# Copyright (c) OpenMMLab. All rights reserved
from __future__ import annotations

from typing import Optional, Tuple

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

    def _better(self, current: float, best: float) -> bool:
        if self.rule == 'greater':
            return current > best + self.min_delta
        return current < best - self.min_delta

    def _get_monitor_value(self, runner) -> Tuple[Optional[float], Optional[str]]:
        output = getattr(runner, 'log_buffer', None)
        if output is None:
            return None, None
        output = runner.log_buffer.output

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

        current, used_key = self._get_monitor_value(runner)
        if current is None:
            runner.logger.warning(
                f'EarlyStoppingHook: monitor="{self.monitor}" not found in '
                f'runner.log_buffer.output this epoch. (looked for '
                f'{self.monitor}/eval_{self.monitor}/val_{self.monitor})')
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
            return

        if self._better(current, self.best_score):
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            runner.logger.info(
                f'EarlyStoppingHook: new best {used_key}={current:.6f} '
                f'at epoch {epoch}.')
            return

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
            self._stop_runner(runner)

