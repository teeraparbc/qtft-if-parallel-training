#!/usr/bin/env python
"""
File:	   ex10_mpi_alltoall.py
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a_size = 2

senddata = (rank+1) * np.arange(size * a_size, dtype=int)
recvdata = np.empty(size*a_size, dtype=int)
comm.Alltoall(senddata, recvdata)

print("process %s sending %s receiving %s " % (rank,senddata,recvdata))
