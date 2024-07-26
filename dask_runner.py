import numpy as np
from astropy.table import Table

import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from matrix_functions import make_matrix

from tqdm import tqdm

def run_row(row):
    tot_matrix = make_matrix(row)
    return tot_matrix, row

try:
    missing_rows = np.load('models_left.npy')
    torun = list(missing_rows)
except(OSError):
    b_table = Table.read('boundaries.fits')
    nrows = len(b_table)
    del b_table
    torun = range(nrows)

print('Starting cluster.')
cluster = SLURMCluster(cores=1,
                       processes=1,
                       memory='48GB',
                       walltime='24:00:00',
                       account='astronomy-dept',
                       job_extra_directives=['--qos=astronomy-dept-b',
                                             '--output=log/%A.out',
                                             '--error=log/%A.err'],
                       nanny=True,
                       worker_extra_args=['--preload','/blue/adamginsburg/richardson.t/research/flux/confusion_dir/matrix_functions.py',
                                          '--lifetime','115m',
                                          '--lifetime-stagger','4m'],
                       scheduler_options={'dashboard_address':':13579'},
)

cluster.adapt(minimum=0,maximum=250)
client = Client(cluster)
print('Starting computation.')
futures = client.map(run_row,torun)
print('Mapping completed.')

for it,future in tqdm(enumerate(futures)):
    result = future.result()
    mat = result[0]
    row = result[1]
    for d,sub_mat in enumerate(mat):
        np.save(f'output/{row}_{d}.npy',sub_mat)

client.close()
cluster.close()
