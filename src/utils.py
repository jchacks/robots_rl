import logging
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
import tqdm
from robots.robot.utils import *

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../")

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
        self.order = {}
        self.hier = []

    def start(self, name="root"):
        self.hier.append(name)
        self.order[tuple(self.hier)] = None
        self.splits[tuple(self.hier)] = time.time()

    def stop(self, remove="root"):
        if remove != self.hier[-1]:
            raise RuntimeError(f"Cannot remove unstarted timer {remove}.")

        name = tuple(self.hier)
        self.times_called[name] += 1
        diff = time.time() - self.splits[name]
        self.diffs[name].append(diff)
        del self.splits[name]
        self.hier.pop()
        return diff

    def mean_diffs(self, name):
        diffs = self.diffs.get(name, None)
        if diffs is None:
            raise KeyError(f"Key '{name}' not found.")
        num = self.times_called[name]
        self.times_called[name] = 0
        return np.mean(diffs) * num

    def wrap(self, name):
        def wrapper(func):
            def inner(*args, **kwargs):
                self.start(name)
                ret = func(*args, **kwargs)
                self.stop(name)
                return ret
            return inner
        return wrapper

    def log_str(self):
        return (
            "\n".join(
                [f"{'-'.join(k)}: {self.mean_diffs(k)}" for k in self.order.keys()]
            )
            + "\n"
        )


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


def discounted(rewards, dones, gamma=0.99):
    discounted = []

    r = rewards[-1]
    for reward, done in zip(rewards[:-1][::-1], dones[::-1]):
        r = reward + gamma * r * (1.0 - done)
        discounted.append(r)
    return np.stack(discounted[::-1])


def gae(rewards, values, masks, gamma=0.99, lmbda=0.95):
    deltas = rewards[:-1] + (gamma * values[1:] * masks) - values[:-1]

    gae = np.zeros(rewards.shape)
    lastgae = 0
    for i in reversed(range(len(deltas))):
        gae[i] = lastgae = deltas[i] + gamma * lmbda * masks[i] * lastgae
    ret = gae + values
    return gae[:-1], ret[:-1]
