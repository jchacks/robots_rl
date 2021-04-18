

class Memory(object):
    def __init__(self, stores:str) -> None:
        self.items = stores.split(',')
        self.data = {k:[] for k in self.items}

    def append(self, **kwargs):
        if set(kwargs.keys()) != set(self.data.keys()):
            raise KeyError("kwargs keys should be the same as data")
        for k,v in kwargs.items():
            self.data[k].append(v)
    
    def __getitem__(self, item):
        return self.data[item]

    def clear(self):
        self.data = {k:[] for k in self.items}
