import logging
import operator
import time
from functools import reduce
from collections import defaultdict, deque
from contextlib import contextmanager
import numba as nb
import numpy as np
import tqdm
from robots.robot.utils import *
import tensorflow as tf

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
    def ctx(self, name=""):
        self.split(name)
        yield
        self.add_diff(name)

    def split(self, name=""):
        self.splits[name] = time.time()

    def diff(self, name=""):
        return time.time() - self.splits[name]

    def add_diff(self, name=""):
        self.times_called[name] += 1
        diff = self.diff(name)
        self.diffs[name].append(diff)
        return diff

    def mean_diffs(self, name=""):
        diffs = self.diffs.get(name, None)
        if diffs is None:
            raise KeyError(f"Key '{name}' not found.")
        num = self.times_called[name]
        self.times_called[name] = 0
        return np.mean(diffs) * num


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
