# Copyright (c) OpenMMLab. All rights reserved
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from mmcv.runner import IterBasedRunner
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Stop training when a monitored metric has stopped improving.

    This hook is designed to work with MMCV's EvalHook/DistEvalHook, which
    writes evaluation metrics into ``runner.log_buffer.output``.
    """

    def __init__(self,
                 monitor: str = 'multi_auc',
                 rule: str = 'greater',
                 patience: int = 10,
                 min_delta: float = 0.0,
                 start_epoch: int = 1,
                 interval: int = 1,
                 check_finite: bool = True):
        if rule not in ('greater', 'less'):
            raise ValueError('rule must be "greater" or "less"')
        if patience < 0:
            raise ValueError('patience must be >= 0')
        if interval < 1:
            raise ValueError('interval must be >= 1')
        if start_epoch < 1:
            raise ValueError('start_epoch must be >= 1 (1-based)')

        self.monitor = monitor
        self.rule = rule
        self.patience = patience
        self.min_delta = float(min_delta)
        self.start_epoch = start_epoch
        self.interval = interval
        self.check_finite = check_finite

        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.num_bad_epochs: int = 0
        self._warned_missing_monitor: bool = False

    def _better(self, current: float, best: float) -> bool:
        if self.rule == 'greater':
            return current > best + self.min_delta
        return current < best - self.min_delta

    def _find_eval_log_buffer(self, runner, epoch: int) -> Optional[dict]:
        """Get the latest eval hook's log buffer snapshot for this epoch."""
        latest = None
        latest_epoch = None
        for hook in getattr(runner, 'hooks', []):
            if not hasattr(hook, 'latest_log_buffer_output'):
                continue
            hook_epoch = getattr(hook, 'latest_eval_epoch', None)
            if hook_epoch is None or hook_epoch != epoch:
                continue
            value = getattr(hook, 'latest_log_buffer_output', None)
            if not isinstance(value, dict):
                continue
            if latest_epoch is None or hook_epoch >= latest_epoch:
                latest = value
                latest_epoch = hook_epoch
        return latest

    def _get_monitor_value(self, runner, epoch: int) -> Tuple[Optional[float], Optional[str]]:
        # Prefer the eval hook snapshot (more reliable than log_buffer which may
        # be cleared/overwritten by logger hooks).
        output = self._find_eval_log_buffer(runner, epoch)
        if output is None:
            output = getattr(runner, 'log_buffer', None)
            output = None if output is None else runner.log_buffer.output

        candidates = (
            self.monitor,
            f'eval_{self.monitor}',
            f'val_{self.monitor}',
        )
        for key in candidates:
            if key in output:
                try:
                    return float(output[key]), key
                except Exception:
                    return None, key

        monitor_lower = self.monitor.lower()
        for key in output.keys():
            if monitor_lower == str(key).lower():
                try:
                    return float(output[key]), key
                except Exception:
                    return None, key
        return None, None

    def _stop_runner(self, runner) -> None:
        # EpochBasedRunner uses `_max_epochs`, IterBasedRunner uses `_max_iters`.
        if isinstance(runner, IterBasedRunner):
            if hasattr(runner, '_max_iters'):
                runner._max_iters = runner.iter + 1
        else:
            if hasattr(runner, '_max_epochs'):
                runner._max_epochs = runner.epoch + 1

        # Some runner implementations check this flag.
        if hasattr(runner, 'should_stop'):
            runner.should_stop = True

    def after_train_epoch(self, runner):
        # Wait until EvalHook has executed for this epoch (it runs in
        # after_train_epoch too). Set this hook's priority to VERY_LOW/LOWEST.
        epoch = runner.epoch + 1  # 1-based for readability
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.interval != 0:
            return

        # In distributed training, only rank 0 has evaluation metrics. We
        # broadcast a stop signal so all ranks stop consistently.
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0

        should_stop = False
        current = None
        used_key = None
        if rank == 0:
            current, used_key = self._get_monitor_value(runner, epoch)
        if current is None:
            if rank == 0 and not self._warned_missing_monitor:
                keys = []
                try:
                    keys = sorted(list(runner.log_buffer.output.keys()))
                except Exception:
                    keys = []
                runner.logger.warning(
                    f'EarlyStoppingHook: monitor="{self.monitor}" not found. '
                    f'Expected one of {self.monitor}/eval_{self.monitor}/'
                    f'val_{self.monitor}. Available keys: {keys}')
                self._warned_missing_monitor = True
            # Still broadcast "continue" in distributed mode.
            if distributed:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                stop_tensor = torch.zeros(1, device=device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
            return

        if self.check_finite:
            # Avoid stopping logic on NaN/Inf values.
            if current != current or current in (float('inf'), float('-inf')):
                runner.logger.warning(
                    f'EarlyStoppingHook: monitor value is not finite '
                    f'({used_key}={current}); skipping.')
                return

        if self.best_score is None:
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            runner.logger.info(
                f'EarlyStoppingHook: init best {used_key}={current:.6f} '
                f'at epoch {epoch}.')
        else:
            if self._better(current, self.best_score):
                self.best_score = current
                self.best_epoch = epoch
                self.num_bad_epochs = 0
                runner.logger.info(
                    f'EarlyStoppingHook: new best {used_key}={current:.6f} '
                    f'at epoch {epoch}.')
            else:
                self.num_bad_epochs += 1
                runner.logger.info(
                    f'EarlyStoppingHook: no improvement in {used_key} '
                    f'(current={current:.6f}, best={self.best_score:.6f} at '
                    f'epoch {self.best_epoch}); bad_epochs={self.num_bad_epochs}/'
                    f'{self.patience}.')

                if self.num_bad_epochs >= self.patience:
                    runner.logger.info(
                        f'EarlyStoppingHook: stopping early at epoch {epoch}. '
                        f'Best {used_key}={self.best_score:.6f} was at epoch '
                        f'{self.best_epoch}.')
                    should_stop = True

        if distributed:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            stop_tensor = torch.tensor(
                [1 if (should_stop and rank == 0) else 0],
                device=device,
                dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            if int(stop_tensor.item()) == 1:
                self._stop_runner(runner)
        else:
            if should_stop:
                self._stop_runner(runner)
