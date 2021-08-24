from mpi4py import MPI
import numpy as np
import time
# from main import EnvEngine 


# size = comm.Get_size()


# class Example(object):
#     def __init__(self) -> None:
#         self.comm =  MPI.COMM_WORLD

#     def update(self, localv):
#         self.comm.
#         self.comm.Allreduce(localv, globalg, op=MPI.SUM)

#     def sync(self):
#         self.comm.Bcast()


# ex = Example()

# ex.update(1)

# print(rank, data)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = np.zeros(100, dtype='i') + rank
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 100], dtype='i')
comm.Gather(sendbuf, recvbuf, root=0)
if rank == 0:
    for i in range(size):
        assert np.allclose(recvbuf[i,:], i)
    print(recvbuf)


def if_rank(rank: int, comm=None):
    def wrapper(func):
        def inner(*args, **kwargs):
            nonlocal comm
            comm = MPI.COMM_WORLD if comm is None else comm
            if comm.Get_rank() == rank:
                ret = func(*args, **kwargs)
                return ret
        return inner
    return wrapper
