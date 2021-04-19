from functools import reduce
import operator
import logging
import tqdm
import numpy as np

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
