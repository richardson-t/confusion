import numpy as np
from astropy.table import Table

import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from matrix_functions import make_matrix

@dask.delayed
def run_row(row):
    tot_matrix = make_matrix(row)
    return tot_matrix, row

b_table = Table.read('boundaries.fits')
nrows = len(b_table)
del b_table

print('Starting cluster.')
cluster = SLURMCluster(cores=1,
                       processes=1,
                       memory='72GB',
                       walltime='96:00:00',
                       account='astronomy-dept',
                       job_extra_directives=['--qos=astronomy-dept-b',
                                             '--output=log/%A.out',
                                             '--error=log/%A.err'],
                       nanny=True,
                       worker_extra_args=['--preload /blue/adamginsburg/richardson.t/research/flux/confusion_dir/matrix_functions.py',
                                          '--lifetime 119m'],
                       scheduler_options={'dashboard_address':':13579'},
)

cluster.adapt(minimum=0,maximum=200)
client = Client(cluster)

print('Starting mapping.')
mapping = client.map(run_row, range(nrows))
delayed = client.gather(mapping)

for res in delayed:
    output = dask.compute(res)
    mat = output[0][0]
    row = output[0][1]
    for d,sub_mat in enumerate(mat):
        np.save(f'output/{row}_{d}.npy',sub_mat)

client.close()
cluster.close()
