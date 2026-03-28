import argparse
import os.path as osp
import warnings

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from model.models import build_classifier


# A dummy function to handle a potential compatibility issue
def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            Shape of input images, e.g., (1, 3, 224, 224).
        num_classes (int):
            Number of classes.
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(N, C, H, W)
    gt_label = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'img': torch.from_numpy(imgs).float(),
        'gt_label': torch.from_numpy(gt_label).long(),
    }
    return mm_inputs

# The main conversion function
def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Export Pytorch model to ONNX model and verify the outputs are same between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct the corresponding dummy input and execute the model.
        opset_version (int): The onnx op-set version to export.
        show (bool): Whether print the computation graph.
        output_file (string): The path to where we store the exported ONNX model.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
    """
    model.cpu().eval()
    
    # Get the number of classes from the model's head
    if hasattr(model, 'head') and hasattr(model.head, 'num_classes'):
        num_classes = model.head.num_classes
    else:
        num_classes=14

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)
    imgs = mm_inputs.pop('img')
    # img_list = [img for img in imgs]

    # Custom Forward Wrapper
    def custom_export_forward(img):
        # 1. Get features from backbone
        # Returns: ([[cnn_tensor, trans_tensor]],)
        feats = model.backbone(img)
        
        # 2. Pass to the head
        cls_score = model.head(feats)
        
        # 3. Final Sigmoid for NIH probabilities
        return torch.sigmoid(cls_score)

    model.forward = custom_export_forward

    # model's forward method for export session
    model.forward = custom_export_forward
    
    # ---- THE CORE ONNX EXPORT CALL ----
    torch.onnx.export(
        model, imgs,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=False,
        verbose=show,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'])
    
    print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        # check by onnx
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # Get PyTorch output using custom forward
        with torch.no_grad():
            pytorch_results = model(imgs).numpy()

        # get onnx output
        sess = ort.InferenceSession(output_file)
        onnx_inputs = {sess.get_inputs()[0].name: imgs.detach().numpy()}
        onnx_results = sess.run(None, onnx_inputs)[0]
        
        # compare results
        np.testing.assert_allclose(
            pytorch_results, onnx_results, rtol=1e-03, atol=1e-05)
        print('The output of ONNXRuntime and PyTorch are the same.')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--shape', type=int, nargs='+', default=[224, 224], help='input image size')
    parser.add_argument('--opset-version', type=int, default=11)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, args.shape[0], args.shape[1])
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    
    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)
    load_checkpoint(classifier, args.checkpoint, map_location='cpu')

    # conver model to onnx file
    pytorch2onnx(
        classifier,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        )