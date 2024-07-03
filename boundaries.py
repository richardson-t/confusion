import numpy as np
from astropy.table import Table

from glob import glob
from tqdm import tqdm

def smooth_bin(data,nbins):
    smooth_data = np.zeros(nbins)
    bin_indices = np.linspace(0,len(data)-1,nbins+1).astype(int)
    for i in range(nbins):
        try:
            bin_data = np.concatenate(data[bin_indices[i]:bin_indices[i+1]])
        except(ValueError):
            bin_data = data[bin_indices[i]:bin_indices[i+1]]
        smooth_data[i] = np.nanmedian(bin_data)
        return smooth_data

def interp_tables(mf,masses,track_dict):    
    # Retrieve the relevant tables (with modifications for interpolation)
    i = np.searchsorted(masses, mf)
    m1 = masses[i-1]; m2 = masses[i]
    interp = Table()
    ev1 = track_dict[m1]
    ev2 = track_dict[m2]
    if history == 'is':
        ev1 = ev1[:1000]
        ev2 = ev2[:1000]
    keys = ev1.keys()
    
    # Interpolate
    frac = (mf - m1) / (m2 - m1)
    for key in keys:
        interp.add_column(ev1[key] * (1. - frac) + ev2[key] * frac)

    return interp

histories = ['is','tc','ca']

#figure out times to sample at for each history
track_masses = np.geomspace(0.2,50,25)
interp_masses = np.geomspace(min(track_masses),max(track_masses),50)

hist_times = []
for history in histories:
    if history == 'is':
        prefix = 'isothermal'
    elif history == 'tc':
        prefix = 'turbulentcore'
    elif history == 'ca':
        prefix = 'competitive'
    
    track_dir = f'protostar_tracks/{prefix}'
    tracks = glob(f'{track_dir}/*.txt')
    tracks.sort()
    track_dict = {}
    for i,mass in enumerate(track_masses):
        track_dict.update({mass:Table.read(tracks[i],format='ascii')})

    timesteps = []

    for it,mass in enumerate(interp_masses):
        evol = interp_tables(mass,track_masses,track_dict)
        valid_times = np.logical_and(mass - evol['Stellar_Mass'] > 0,np.isfinite(evol['Stellar_Temperature']))
        evol = evol[valid_times]
        tstar = evol['Stellar_Temperature']
        lstar = evol['Intrinsic_Lum']
        time = evol['Time']
        timesteps.extend([t for t in time])
    hist_times.append(timesteps)
    
time_endpoints = []
for it,arr in enumerate(hist_times):
    if histories[it] == 'is':
        time_edges = 10**np.histogram_bin_edges(np.log10(arr),bins='auto')
    else:
        time_edges = np.histogram_bin_edges(arr,bins='auto')
    mid_endpoints = []
    for i in range(len(time_edges)):
        for j in range(i+1,len(time_edges)):
            mid_endpoints.append([time_edges[i],time_edges[j]])
    time_endpoints.append(mid_endpoints)
            
mass_edges = [0.2, 0.45, 1, 2, 5, 10, 23, 50]
mass_endpoints = []
for i in range(len(mass_edges)):
    for j in range(i+1,len(mass_edges)):
        mass_endpoints.append([mass_edges[i],mass_edges[j]])

n_effs = 8
effs_torun = []
for i in range(n_effs):
    for j in range(i+1,n_effs+1):
        effs_torun.append([i,j])

distances = [np.nan,0.1,0.5,1,5,10]

row = 0
h_dict = {}
m_dict = {}
t_dict = {}
e_dict = {}
#d_dict = {}
print('Building boundary table...')
boundary_table = Table()
for it,h in enumerate(histories):
    for m_pair in mass_endpoints:
        for t_pair in time_endpoints[it]:
            for e_list in effs_torun:
                #for dist in distances:
                h_dict.update({row:h})
                m_dict.update({row:m_pair})
                t_dict.update({row:t_pair})
                e_dict.update({row:e_list})
                #d_dict.update({row:dist})
                row += 1

boundary_table = Table([[value for value in h_dict.values()],
                        [value for value in m_dict.values()],
                        [value for value in t_dict.values()],
                        [value for value in e_dict.values()]],
                        #[value for value in d_dict.values()]],
                        names=['AH','Mf','Age','SFE']#,'Detectability']
)
boundary_table.write('confusion_dir/boundaries.fits',overwrite=True)

'''
format:
AH | Final mass | Time | SFE | Detectability | Matrix
'''
