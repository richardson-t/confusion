import numpy as np
from astropy.table import Table
from astropy import units as u
from sedfitter.sed import SEDCube

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable,get_cmap
from matplotlib.collections import LineCollection

from user_functions import fullgrid_seedling,nearby_models
from util import geo_inc

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
    
def make_matrix(stages,classes,detectable=None,thresh=None):
    tec = False
    if detectable is not None:
        tec = True
        stages = stages[detectable]
        classes = classes[detectable]
        
    stage_nums = [0,1,2,3,-1]
    class_nums = [0,1,2,3,4,-1]
    matrix = np.zeros((len(stage_nums),len(class_nums)))
    total = len(stages)

    for i in stage_nums:
        stage_cut = stages == i
        cut_class = classes[stage_cut]
        for j in class_nums:
            matrix[i,j] = len(cut_class[cut_class == j])

    stage_counts = np.sum(matrix,axis=-1)
    class_counts = np.sum(matrix,axis=0)

    matrix *= 100 / total
    return matrix,stage_counts,class_counts,tec

def plot_matrix(matrix,stage_counts,class_counts,tec):
    p_matrix = np.append(matrix,np.nan*np.ones(len(stage_counts))[:,None],axis=-1)
    p_matrix = np.append(p_matrix,np.nan*np.ones(len(class_counts)+1)[None,:],axis=0)
    text_matrix = np.append(matrix,stage_counts[:,None],axis=-1)
    text_matrix = np.append(text_matrix,np.append(class_counts,np.nan)[None,:],axis=0)

    y,x = p_matrix.shape
    xed,yed = np.arange(0,x+1),np.arange(0,y+1)
    XED,YED = np.meshgrid(xed,yed,indexing='ij')
    xav = (xed[1:] + xed[:-1]) / 2
    yav = (yed[1:] + yed[:-1]) / 2; yav = yav[::-1]

    stage_names = ['Stage 0','Stage I','Stage II','Stage III','No Stage','# with Class']
    class_names = ['Class 0','Class I','Flat','Class II','Class III','No Class','# with Stage']
    norm = LogNorm(0.1,np.max(matrix))

    plt.figure(figsize=(11,7.2))
    plt.pcolormesh(XED,YED,p_matrix.T[:,::-1],norm=norm,cmap=cmap,alpha=0.3)
    cb = plt.colorbar()
    cb.set_label('% All',fontsize='large')
    plt.xticks(ticks=xav,labels=class_names,fontsize='large')
    plt.gca().tick_params(top=True,labeltop=True,bottom=False,labelbottom=False)
    plt.yticks(ticks=yav,labels=stage_names,fontsize='large')
    for i in range(len(xav)):
        for j in range(len(yav)):
            if i == len(xav)-1 or j == len(yav)-1:
                if i == len(xav)-1 and j == len(yav)-1:
                    pass
                else:
                    plt.text(xav[i],yav[j],f'{int(text_matrix[j,i])}',
                             ha='center',va='center',fontsize='large')
            else:
                plt.text(xav[i],yav[j],f'{np.round(text_matrix[j,i],2)}%',
                         ha='center',va='center',fontsize='large')
    if tec:
        plt.title(f'{history.upper()}, ALMA Detectable at {thresh}',fontsize='large')
        plt.savefig(f'../matrices/{history}_{dist}_detectable.pdf',dpi=300,bbox_inches='tight')
    else:
        plt.title(f'{history.upper()}, All',fontsize='large')
        plt.savefig(f'../matrices/{history}_{dist}.pdf',dpi=300,bbox_inches='tight')

modeldir = '/blue/adamginsburg/richardson.t/research/flux/r+24_models-1.2'
geometries = ['s-p-hmi','s-p-smi','s-pbhmi','s-pbsmi','s-u-hmi','s-u-smi',     
              's-ubhmi','s-ubsmi','spu-hmi','spu-smi','spubhmi','spubsmi']
history = 'tc'
dist = 'quant'

if history == 'is':
    prefix = 'isothermal'
    cmap = get_cmap('viridis')
elif history == 'tc':
    prefix = 'turbulentcore'
    cmap = get_cmap('plasma')
elif history == 'ca':
    prefix = 'competitive'
    cmap = get_cmap('cividis')

print('Loading models...')
if dist == 'log':
    fulldat, valid_models, valid_pars, tree = fullgrid_seedling(geometries,
                                                                dist=dist,
                                                                modeldir=modeldir)
    norms = None
    transformer = None
elif dist == 'metric':
    fulldat, valid_models, valid_pars, norms = fullgrid_seedling(geometries,
                                                                 dist=dist,
                                                                 modeldir=modeldir)
    tree = None
    transformer = None
elif dist == 'quant':
    fulldat, valid_models, valid_pars, tree, transformer = fullgrid_seedling(geometries,
                                                                             dist=dist,
                                                                             modeldir=modeldir)
    norms = None

track_masses = np.geomspace(0.2,50,25)
track_dir = f'../protostar_tracks/{prefix}'
track_dict = {}
for i,mass in enumerate(track_masses):
    track_dict.update({mass:Table.read(f'{track_dir}/protostellar_evolution_m={mass}.txt',
                                       format='ascii')})

flux_ap = 10
effs = [1/6, 1/4, 1/3, 1/2, 2/3]

all_geos = []
all_names = []

print('Starting model association...')
masses = np.geomspace(min(track_masses),max(track_masses),50)
for it,mass in tqdm(enumerate(masses)):
    evol = interp_tables(mass,track_masses,track_dict)
    valid_times = np.logical_and(mass - evol['Stellar_Mass'] > 0,np.isfinite(evol['Stellar_Temperature']))
    evol = evol[valid_times]
    tstar = evol['Stellar_Temperature']
    lstar = evol['Intrinsic_Lum']
    for eff in effs:
        mcore = (mass-evol['Stellar_Mass']) / eff

        for step in range(len(evol)):
            names = nearby_models(tstar[step],lstar[step],mcore[step],
                                  dist=dist,fulldat=fulldat,valid_pars=valid_pars,
                                  tree=tree,norms=norms,transformer=transformer)
            for name in names:
                this_geo, this_model = name.split('/')
                #if this_model not in all_names:
                all_geos.append(this_geo)
                all_names.append(this_model)

all_geos = np.array(all_geos)
all_names = np.array(all_names)                

stages = []
classes = []

detectable = []
thresh = 1*u.mJy
distance = 5*u.kpc

sed_dict = {g:SEDCube.read(f'{modeldir}/{g}/flux.fits') for g in geometries}

print('Retrieving model information:')
for g in tqdm(geometries):
    indices = []
    stats = Table.read(f'{modeldir}/{g}/info.fits')
    idx = np.arange(0,len(stats))
    
    #find the stats for each model once and then map those out
    these_names, inv = np.unique(all_names[all_geos == g],
                                 return_inverse=True)
    for name in these_names:
        where_model = [n[:8] == name for n in stats['Model Name']]
        indices.append(idx[where_model])
    indices = np.array(indices)
    indices = np.ravel(indices[inv])

    stages.extend([value for value in stats['Stage'][indices]])
    classes.extend([value for value in stats['Class'][indices]])
    
    #is the 1mm flux detectable with ALMA (i.e. > 1 mJy at 5 kpc in a 2000 AU aperture)?
    d = sed_dict[g].val[indices,6,24] > thresh * (distance / u.kpc)**2
    detectable.extend([value for value in d])

stages = np.array(stages)
classes = np.array(classes)
detectable = np.array(detectable)

print('Building matrices:')
args = make_matrix(stages,classes)
plot_matrix(*args)
args = make_matrix(stages,classes,detectable=detectable)
plot_matrix(*args)
print('Done.')
