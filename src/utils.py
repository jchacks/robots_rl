import logging
import operator
import time
from functools import reduce
from collections import defaultdict, deque

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
        self.times = deque(maxlen=maxlen)
        self.splits = defaultdict(lambda: deque(maxlen=maxlen))

    def block(self):
        start = time.time()
        yield
        end = time.time()
        self.times.append(end - start)

    def time(self, func):
        def inner(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            self.times.append(end - start)
            return res
        return inner

    def split(self, name=""):
        self.splits[name].append(time.time())

    def since(self, name=""):
        return time.time() - self.splits[name][-1]

    def last_diff(self, name=""):
        if len(self.splits[name]) > 1:
            return self.splits[name][-1] - self.splits[name][-2]
        else:
            return np.nan

    def mean_diff(self, name=""):
        times = np.array(self.splits[name])
        return np.mean(times[1:] - times[:-1])


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


class Memory(object):
    def __init__(self, stores: str) -> None:
        self.items = stores.split(',')
        self.data = {k: [] for k in self.items}

    def append(self, **kwargs):
        if set(kwargs.keys()) != set(self.data.keys()):
            raise KeyError("kwargs keys should be the same as data")
        for k, v in kwargs.items():
            self.data[k].append(v)

    def __getitem__(self, item):
        return self.data[item]

    def clear(self):
        self.data = {k: [] for k in self.items}


def discounted(rewards, dones, last_value, gamma=0.99):
    discounted = []
    r = last_value
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.0 - done)
        discounted.append(r)
    return np.concatenate(discounted[::-1])
