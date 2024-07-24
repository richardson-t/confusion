import numpy as np
from os import system

import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from matrix_functions import make_matrix

@dask.delayed
def run_row(row):
    tot_matrix = make_matrix(row)
    return tot_matrix, row

cluster = SLURMCluster(cores=1,
                       processes=1,
                       memory='72GB',
                       walltime='96:00:00',
                       account='astronomy-dept',
                       job_extra_directives=['--qos=astronomy-dept-b',
                                             '--output=log/dask/%A.out',
                                             '--error=log/dask/%A.err'],
                       nanny=True,
                       worker_extra_args=['--preload /blue/adamginsburg/richardson.t/research/flux/confusion_dir/matrix_functions.py'],
)

cluster.adapt(minimum=0,maximum=200)
client = Client(cluster)
mapping = client.map(run_row, range(2))
delayed = client.gather(mapping)

for res in enumerate(delayed):
    mat, row = dask.compute(res)
    for d,sub_mat in enumerate(mat):
        np.save(f'output/dask/{row}_{d}.npy',sub_mat)

client.close()
cluster.close()
