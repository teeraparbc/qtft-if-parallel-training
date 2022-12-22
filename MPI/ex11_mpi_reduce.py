#!/usr/bin/env python
"""
File:	   ex11_mpi_reduce.py
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
t_start = MPI.Wtime()

# this array lives on each processor
data = np.zeros(5)
for i in range(comm.rank, len(data), comm.size):
    # set data in each array that is different for each processor
    data[i] = i

comm.Barrier()
# print out the data arrays for each processor
for r in range(size):
    if rank == r:
        print('rank %i has data' % rank, data)
comm.Barrier()

# the 'totals' array will hold the sum of each 'data' array
if rank==0:
    # only processor 0 will actually get the data
    totals = np.zeros_like(data)
else:
    totals = None

# use MPI to get the totals 
comm.Reduce([data, MPI.DOUBLE], [totals, MPI.DOUBLE], op = MPI.SUM, root = 0)

if rank==0:
    print('The total sum is', totals)

comm.Barrier()
t_diff = MPI.Wtime() - t_start
if comm.rank==0:
    print('Total time used ', t_diff)
