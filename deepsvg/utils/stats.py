import datetime
from collections import defaultdict
from collections import deque

import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricTracker(object):
    """
    Track a series of values and provide access both the global series and a buffer that can be reset.
    """

    def __init__(self):
        self.buffer_values = []
        self.all_values = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.buffer_values.append(value)
        self.all_values.append(value)
        self.count += 1
        self.total += value

    def median(self, buffer=True):
        if buffer:
            x = self.buffer_values
        else:
            x = self.all_values

        return torch.tensor(x).median().item()

    def avg(self, buffer=True):
        if buffer:
            x = self.buffer_values
        else:
            x = self.all_values

        return torch.tensor(x).mean().item()

    def reset_buffer(self):
        self.buffer_values = []


class Stats:
    def __init__(self, num_steps=None, num_epochs=None, steps_per_epoch=None, stats_to_print=None):
        self.step = self.epoch = 0

        if num_steps is not None:
            self.num_steps = num_steps
        else:
            self.num_steps = num_epochs * steps_per_epoch

        self.stats = {
            k: defaultdict(MetricTracker) for k in stats_to_print.keys()
        }
        self.stats_to_print = {k: set(v) for k, v in stats_to_print.items()}

    def to_dict(self):
        return self.__dict__

    def load_dict(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)

    def update(self, split, step, epoch, dict):
        self.step = step
        self.epoch = epoch

        for k, v in dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.stats[split][k].update(v)

    def update_stats_to_print(self, split, stats_to_print):
        self.stats_to_print[split].update(stats_to_print)

    def reset_stats_to_print(self):
        for split in self.stats_to_print.keys():
            self.stats_to_print[split].clear()

    def get_summary(self, split):

        if split == "train":
            completion_pct = self.step / self.num_steps * 100
            eta_seconds = self.stats[split].get("time").avg(buffer=False) * (self.num_steps - self.step)
            eta_string = datetime.timedelta(seconds=int(eta_seconds))

            s = "[{}/{}, {:.1f}%] eta: {}, ".format(self.step, self.num_steps, completion_pct, eta_string)
        else:
            s = f"[{split.upper()}, epoch {self.epoch + 1}] "

        return s + ", ".join(
            f"{stat}: {self.stats[split].get(stat).median():.4f}"
            for stat in self.stats_to_print[split]
            if self.stats[split].get(stat) is not None
        )

    def write_tensorboard(self, summary_writer, split, reset_buffers_after_write=True):
        print(f"Writing split={split} to tensorboard. reset_buffers={reset_buffers_after_write}. step={self.step}")
        summary_writer.add_scalar(f"epoch/{split}", self.epoch + 1, self.step)

        for stat in self.stats_to_print[split]:
            summary_writer.add_scalar(
                f"{stat}" if split in stat else f"{split}_{stat}",
                self.stats[split].get(stat).median(),
                self.step,
            )

        if reset_buffers_after_write:
            self.reset_buffers()

    def is_best(self):
        return True

    def reset_buffers(self):
        for split in self.stats.keys():
            for stat in self.stats[split].keys():
                self.stats[split][stat].reset_buffer()
