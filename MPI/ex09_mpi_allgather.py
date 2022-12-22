#!/usr/bin/env python
"""
File:	   ex09_mpi_allgather.py
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if comm.rank == 0:
    print("-"*78)
    print(" Running on %d cores" % comm.size)
    print("-"*78)

my_N = 4
N = my_N * comm.size
    
if comm.rank == 0:
    A = np.arange(N, dtype=np.float64)
else:
    A = np.empty(N, dtype=np.float64)

my_A = np.empty(my_N, dtype=np.float64)

# Scatter data into my_A arrays
comm.Scatter( [A, MPI.DOUBLE], [my_A, MPI.DOUBLE])

if comm.rank == 0:
    print("After Scatter:")

for r in range(comm.size):
    if comm.rank == r:
        print("[%d] %s" % (comm.rank, my_A))

comm.Barrier()

# Everybody is multiplying by 2
my_A *= 2

comm.Barrier()

#if comm.rank == 0:
comm.Allgather([my_A, MPI.DOUBLE], [A, MPI.DOUBLE])
print("After AllGather:")

print("[%d] %s" % (comm.rank, A))
