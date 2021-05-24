import logging
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
import tqdm
from robots.robot.utils import *

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + '/../')

# Action selection lists
TURNING = [Turn.NONE, Turn.LEFT, Turn.RIGHT]
MOVING = [Move.NONE, Move.FORWARD, Move.BACK]


def cast(dtype):
    def wrap(function):
        def inner(*args, **kwargs):
            return tf.cast(function(*args, **kwargs), dtype)
        return inner
    return wrap


class Timer(object):
    def __init__(self, maxlen=10):
        self.splits = {}
        self.diffs = defaultdict(lambda: deque(maxlen=maxlen))
        self.times_called = defaultdict(int)

    @contextmanager
    def ctx(self, name="root"):
        self.split(name)
        yield
        self.add_diff(name)

    def start(self, name="root"):
        self.splits[name] = time.time()

    def diff(self, name="root"):
        return time.time() - self.splits[name]

    def stop(self, name="root"):
        self.times_called[name] += 1
        diff = self.diff(name)
        self.diffs[name].append(diff)
        del self.splits[name]
        return diff

    def mean_diffs(self, name="root"):
        diffs = self.diffs.get(name, None)
        if diffs is None:
            raise KeyError(f"Key '{name}' not found.")
        num = self.times_called[name]
        self.times_called[name] = 0
        return np.mean(diffs) * num

    def log_str(self):
        return '\n'.join([f"{k}: {self.mean_diffs(k)}" for k in self.diffs.keys()]) + '\n'


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def make_logger():
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    logger.addHandler(TqdmLoggingHandler())
    return logger


def discounted(rewards, dones, last_value, gamma=0.99):
    discounted = []
    r = last_value
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.0 - done)
        discounted.append(r)
    return np.concatenate(discounted[::-1])
