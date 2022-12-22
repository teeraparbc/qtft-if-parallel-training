#!/usr/bin/env python
"""
File:	   ex12_mpi_reduce_scatter.py
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
t_start = MPI.Wtime()

# this array lives on each processor
data = np.zeros(10)
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
    totals = np.zeros(5)
else:
    totals = np.zeros(5)

# use MPI to get the totals
# Reduce_scatter_block will also work without recvcounts
comm.Reduce_scatter([data, MPI.DOUBLE], [totals, MPI.DOUBLE], recvcounts=[5, 5],
                    op = MPI.SUM)

#if rank==0:
print('From process ', rank, ': the total sum is', totals)

comm.Barrier()
t_diff = MPI.Wtime() - t_start
if comm.rank==0:
    print('Total time used ', t_diff)
