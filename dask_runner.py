import numpy as np
from astropy.table import Table

import dask
import dask.bag as db
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from matrix_functions import make_matrix

from tqdm import tqdm

def run_row(row):
    tot_matrix = make_matrix(row)
    return tot_matrix, row

def write_matrix(future):
    res = future.result()
    mat = res[0]
    row = res[1]
    for d,sub_mat in enumerate(mat):
        np.save(f'output/{row}_{d}.npy',sub_mat)
    return 0

b_table = Table.read('boundaries.fits')
nrows = len(b_table)
del b_table

print('Starting cluster.')
cluster = SLURMCluster(cores=1,
                       processes=1,
                       memory='96GB',
                       walltime='96:00:00',
                       account='astronomy-dept',
                       job_extra_directives=['--qos=astronomy-dept-b',
                                             '--output=log/%A.out',
                                             '--error=log/%A.err'],
                       nanny=True,
                       worker_extra_args=['--preload /blue/adamginsburg/richardson.t/research/flux/confusion_dir/matrix_functions.py',],
                       scheduler_options={'dashboard_address':':13579'},
)

cluster.adapt(minimum=0,maximum=200)
client = Client(cluster)
print('Starting mapping.')
futures = client.map(run_row, range(nrows))
print('Mapping complete.')

for it,future in tqdm(enumerate(futures)):
    result = future.result()
    mat = result[0]
    row = result[1]
    for d,sub_mat in enumerate(mat):
        np.save(f'output/{row}_{d}.npy',sub_mat)

client.close()
cluster.close()
