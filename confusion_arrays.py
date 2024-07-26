import numpy as np
from astropy.table import Table
from astropy import units as u
from sedfitter.sed import SEDCube
from argparse import ArgumentParser

from user_functions import fullgrid_seedling,nearby_models

from glob import glob
from tqdm import tqdm

#from os import chdir
#chdir('/blue/adamginsburg/richardson.t/research/flux')

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
    
def make_matrix(stages,classes):        
    stage_nums = [0,1,2,3,-1]
    class_nums = [0,1,2,3,4,-1]
    matrix = np.zeros((len(stage_nums),len(class_nums)))

    for i in stage_nums:
        stage_cut = stages == i
        cut_class = classes[stage_cut]
        for j in class_nums:
            matrix[i,j] = len(cut_class[cut_class == j])

    return matrix

parser = ArgumentParser()
parser.add_argument('-r','--row',action='store',dest='row',
                    help='row in boundary table') 
args = parser.parse_args()

b_table = Table.read('boundaries.fits')
job_props = b_table[int(args.row)]
del b_table

modeldir = '../r+24_models-1.2'
geometries = ['s-p-hmi','s-p-smi','s-pbhmi','s-pbsmi','s-u-hmi','s-u-smi',     
              's-ubhmi','s-ubsmi','spu-hmi','spu-smi','spubhmi','spubsmi']
dist = 'quant'
mass_ap = 10

effs = [1/6., 1/4., 1/3., 1/2., 2/3., 1., 2., 3.]
history = job_props['AH']
effs_torun = effs[job_props['SFE'][0]:job_props['SFE'][1]]
#distance = job_props['Detectability'] * u.kpc

print('Loading models...')
if dist == 'none':
    fulldat, valid_models, valid_pars, tree = fullgrid_seedling(geometries,dist=dist,
                                                                modeldir=modeldir)
    norms = None
    transformer = None
elif dist == 'metric':
    fulldat, valid_models, valid_pars, norms = fullgrid_seedling(geometries,dist=dist,
                                                                 modeldir=modeldir)
    tree = None
    transformer = None
elif dist == 'quant':
    fulldat, valid_models, valid_pars, tree, transformer = fullgrid_seedling(geometries,dist=dist,
                                                                             modeldir=modeldir)
    norms = None

#figure out times to sample at for each history
if history == 'is':
    prefix = 'isothermal'
elif history == 'tc':
    prefix = 'turbulentcore'
elif history == 'ca':
    prefix = 'competitive'

track_masses = np.geomspace(0.2,50,25)
trackdir = f'../protostar_tracks/{prefix}'
tracks = glob(f'{trackdir}/*.txt')
tracks.sort()
track_dict = {}
for i,mass in enumerate(track_masses):
    track_dict.update({mass:Table.read(tracks[i],format='ascii')})
    
all_geos = []
all_names = []
model_times = []

interp_masses = np.geomspace(min(track_masses),max(track_masses),50)
mass_cut = np.logical_and(interp_masses >= job_props['Mf'][0],interp_masses <= job_props['Mf'][1])
interp_masses = interp_masses[mass_cut]

for it,mass in tqdm(enumerate(interp_masses)):
    evol = interp_tables(mass,track_masses,track_dict)
    valid_times = np.logical_and(mass - evol['Stellar_Mass'] > 0,np.isfinite(evol['Stellar_Temperature']))
    evol = evol[valid_times]
    tstar = evol['Stellar_Temperature']
    lstar = evol['Intrinsic_Lum']

    for eff in effs_torun:
        mcore = (mass-evol['Stellar_Mass']) / eff

        for step in range(len(evol)):
            names = nearby_models(tstar[step],lstar[step],mcore[step],
                                  dist=dist,fulldat=fulldat,valid_pars=valid_pars,
                                  tree=tree,norms=norms,transformer=transformer)
            for name in names:
                this_geo, this_model = name.split('/')
                #if this_model not in all_names: #double count
                all_geos.append(this_geo)
                all_names.append(this_model)
            model_times.extend(evol['Time'][step] * np.ones(len(names)))
            
all_geos = np.array(all_geos)
all_names = np.array(all_names)
model_times = np.array(model_times)

time_cut = np.logical_and(model_times >= job_props['Age'][0],model_times <= job_props['Age'][1])
if np.any(time_cut):
    all_geos = all_geos[time_cut]
    all_names = all_names[time_cut]

    stages = []
    classes = []
    distances = np.array([np.nan,0.1,0.5,1,5,10]) * u.kpc
    detectable = []

    included_geos = np.unique(all_geos)
    sed_dict = {g:SEDCube.read(f'{modeldir}/{g}/flux.fits') for g in included_geos}

    print('Retrieving model information...')
    for g in tqdm(included_geos):
        geo_cut = all_geos == g
        indices = []
        stats = Table.read(f'{modeldir}/{g}/info.fits')
        idx = np.arange(0,len(stats))

        these_names, inv = np.unique(all_names[geo_cut],
                                     return_inverse=True)
        for name in these_names:
            where_model = [n[:8] == name for n in stats['Model Name']]
            indices.append(idx[where_model])
        indices = np.array(indices)
        indices = np.ravel(indices[inv])
        
        stages.extend([value for value in stats['Stage'][indices]])
        classes.extend([value for value in stats['Class'][indices]])

        mid_detect = []
        for d in distances:
            if not np.isfinite(d):
                mid_detect.append(np.ones(len(indices)))
            else:
                det = sed_dict[g].val[indices,6,24] > 1 * u.mJy * (d / u.kpc)**2
                mid_detect.append([value for value in det])
        detectable.append(mid_detect)

    stages = np.array(stages)
    classes = np.array(classes)
    detectable = np.concatenate(detectable,axis=-1).astype(bool)

    for it,det_array in enumerate(detectable):
        matrix = make_matrix(stages[det_array],classes[det_array])
        np.save(f'output/{args.row}_{it}.npy', matrix)

else:
    matrix = np.zeros((5,6))
    for it in range(6):
        np.save(f'output/{args.row}_{it}.npy',matrix)
