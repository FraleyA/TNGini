import sys
sys.path.append('/home/fraley.a/packages')

import numpy as np
import matplotlib.pyplot as plt
import h5py
import illustris_python as il
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import os
import glob
from scipy.spatial import cKDTree


h = 0.6774 # Hubble normalization constant used by TNG


def get_subhalo_position(subbox_path, subhalo_id, snap_num):
    """ Return the comoving position of the subhalo in the given snapshot per h (ckpc/h). """
    with h5py.File(subbox_path, 'r') as hf:
        subhalo = np.where(hf['SubhaloIDs'][:] == subhalo_id)[0]
        subhalo_position = hf['SubhaloPos'][subhalo, snap_num, :]
        return subhalo_position[0]


def get_scale_factor(subbox_path, snap_num):
    """ Return the scale factor of a given subbox snapshot. """
    with h5py.File(subbox_path, 'r') as hf:
        return hf['SubboxScaleFac'][snap_num]

    
def stellar_age(subbox_path, snap_num, GFM_StellarFormationTime, H0=67.74, Om0=0.3089, Ob0=0.0486):
    """ Use astropy FlatLambdaCDM with TNG cosmological parameters to obtain stellar formation time.
        GFM_StellarFormationTime: The scale factor corresponding to the when the stellar particle is born.
        Om0: Density of non-relativistic matter in units of critical density at z=0.
        Ob0: Density of baryonic in units of critical density at z=0.
    """
    cosmology = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0, Ob0=Ob0)
    
    snap_scale_factor = get_scale_factor(subbox_path, snap_num)
    snap_redshift = (1 / snap_scale_factor) - 1
    snap_time = cosmology.age(snap_redshift).to(u.yr).value
    
    stellar_redshift = (1 / GFM_StellarFormationTime) - 1
    stellar_formation_time = cosmology.age(stellar_redshift).to(u.yr).value
    
    return snap_time - stellar_formation_time


def get_smoothing_length(positions):
    """ StellarHsml is unfortunately not present in the subbox snapshot data. 
        Use scipy.spatial.cKDTree for a quick nearest neighbor search.
        positions: (N, 3) use the standard positions array from snapshot data converted to pc.
    """
    
    # Construct the cKDTree
    tree = cKDTree(positions)
    
    # Query for the 32nd nearest neighbor
    k = 32 + 1
    distances, indices = tree.query(positions, k=k)
    
    # Make my own StellarHsml field
    StellarHsml = distances[:, 32]
    
    return StellarHsml


def star_formation_rate(initial_mass, time_interval=1e7):
    """ Calculate SFRs [Msun / yr] for stars younger than 10 Myr by dividing the star's initial mass by the time inverval.
        SFR assumed to be constant for the past 10 Myr, Rodriguez-Gomez.
    """
    return initial_mass / time_interval


def stellar_data(basePath, subboxNum, subboxSnapNum, id, SSLSnapNum=99, maxDistance=200, maxSmoothingLength=800):
    """ Retrieve stellar particle data for use in the SKIRT radiative transfer Monte Carlo simulation. 
        id: SubhaloIDs id from the subbox subhalo list (i.e. subfindID).
        maxDistance (ckpc/h): maximum distance a particle can be from the subhalo's position.
        maxSmoothingLength: Cap on the smoothing length to prevent over-smoothing.
        NOTE: relative_position is the vector from the subhalo center to the star particle, I am choosing
        the subhalo center to be the origin of the simulated image.
    """
    
    # All of the directories, and files needed to process the data.
    snap_dir = basePath + f'/subbox{subboxNum}_snapdirs/snap_subbox{subboxNum}_{subboxSnapNum}'
    subbox_path = basePath + f'/postprocessing/SubboxSubhaloList/subbox{subboxNum}_{SSLSnapNum}.hdf5'
    
    snap_file = f'snap_subbox{subboxNum}_{subboxSnapNum}.*.hdf5'
    snap_file_pattern = os.path.join(snap_dir, snap_file)
    subbox_snap_chunks = sorted(glob.glob(snap_file_pattern))
    
    # Arrays to store the related data.
    positions = [] # will be converted into an (N, 3) array
    x = np.array([], dtype=float)
    y = np.array([], dtype=float)
    z = np.array([], dtype=float)
    initial_mass = np.array([], dtype=float)
    metallicity = np.array([], dtype=float)
    age = np.array([], dtype=float)
    
    # Grab the scale factor for physical unit conversions
    scale_factor = get_scale_factor(subbox_path, subboxSnapNum)
    
    for file in subbox_snap_chunks:
        chunk_num = file.split('.')[2]
        print(f'Processing chunk {chunk_num}...\n')
        with h5py.File(file, 'r') as hf:
            #print(hf.keys())

            # Get star particles in the snapshot file chunk.
            if 'PartType4' in hf.keys():
                stellar_sources = hf['PartType4']
                #print(stellar_sources.keys())

                # Define subhalo position and stellar particle positions within the subbox snapshot chunk.
                subhalo_position = get_subhalo_position(subbox_path, id, subboxSnapNum) # ckpc/h
                star_positions = stellar_sources['Coordinates'][:] # ckpc/h

                # Calculate the relative position between each star particle in the chunk, and the subhalo.
                relative_positions = star_positions - subhalo_position
                relative_distances = np.linalg.norm(relative_positions, axis=1) # kpc

                # Only consider stellar particles that are < 200 kpc from the subhalo position.
                #print(relative_distance)
                mask = relative_distances < maxDistance
                relative_positions = relative_positions[mask] * (scale_factor / h) # kpc (physical)
                #print(relative_positions.shape)
                
                if len(relative_positions) > 0:
                    #print(star_positions)
                    print(f'\nThere are {len(relative_positions)} star particles within {maxDistance} kpc of the subhalo in chunk {chunk_num}.')
                    print('\n')

                    ### All of the data required for Bruzual & Charlot Simple Stellar Population ###

                    # Initial stellar masses
                    initial_masses = stellar_sources['GFM_InitialMass'][:][mask] # Msun
                    #print(initial_masses)

                    # Metallicity
                    metallicities = stellar_sources['GFM_Metallicity'][:][mask]
                    #print(metallicities)

                    # Calculate stellar ages in Myr based on 'GFM_StellarFormationTime'.
                    GFM_StellarFormationTimes = stellar_sources['GFM_StellarFormationTime'][:][mask] # scale factor when stellar particle formed
                    stellar_ages = stellar_age(subbox_path, subboxSnapNum, GFM_StellarFormationTimes)
                    #print(stellar_ages)
                    
                    # Append arrays to a list for smoothing length calculation.
                    positions.append(relative_positions)
                    
                    xpos = relative_positions[:, 0] * 1e3 # in pc
                    ypos = relative_positions[:, 1] * 1e3 # in pc
                    zpos = relative_positions[:, 2] * 1e3 # in pc
                    
                    initial_masses = (initial_masses * 1e10) / h # convert to Msun (physical)
                    
                    x = np.append(x, xpos)
                    y = np.append(y, ypos) 
                    z = np.append(z, zpos) 
                    initial_mass = np.append(initial_mass, initial_masses)
                    metallicity = np.append(metallicity, metallicities)
                    age = np.append(age, stellar_ages)
                else:
                    continue
            else:
                print(f'Chunk {chunk_num} does not have field "PartType4".\n')
            
    # Finally convert positions to a (N, 3) numpy array for the smoothing length calculation
    positions = np.concatenate(positions, axis=0)
    smoothing_length = get_smoothing_length(positions)
    smoothing_length = np.minimum(smoothing_length, maxSmoothingLength) # replaces all values of h > 800 pc with 800 pc
    
    return x, y, z, smoothing_length, initial_mass, metallicity, age


def write_stellar_source_file(basePath, subboxNum, subboxSnapNum, id, SED_model, path='/home/fraley.a/merger_morphology'):
    """ SED_model must be specified, i.e. BruzualCharlot or MAPPINGS. """
    
    # Obtain the stellar source data
    x, y, z, smoothing_length, initial_mass, metallicity, age = stellar_data(basePath, subboxNum, subboxSnapNum, id)
    
    if SED_model == 'BruzualCharlot':
        file_name = path + '/old_stellar_population.txt'
        
        with open(file_name, 'w') as f:
            f.write(f'# My Stellar Sources File Written From Python - Merger in Subbox Snapshot {subboxSnapNum} Related to {id} in the Subbox Subhalo List\n')
            f.write('# column 1: position x (pc)\n')
            f.write('# column 2: position y (pc)\n')
            f.write('# column 3: position z (pc)\n')
            f.write('# column 4: smoothing length (pc)\n')
            f.write('# column 5: initial mass (Msun)\n')
            f.write('# column 6: metallicity (1)\n')
            f.write('# column 7: age (yr)\n')
        
            for i in range(len(x)):
                # Bruzual & Charlot SED model
                if age[i] > 1e7:
                    f.write(f'{x[i]} {y[i]} {z[i]} {smoothing_length[i]} {initial_mass[i]} {metallicity[i]} {age[i]}\n')
    
    if SED_model == 'MAPPINGS':
        file_name = path + '/young_stellar_population.txt'
        
        with open(file_name, 'w') as f:
            f.write(f'# My Stellar Sources File Written From Python - Merger in Subbox Snapshot {subboxSnapNum} Related to {id} in the Subbox Subhalo List\n')
            f.write('# column 1: position x (pc)\n')
            f.write('# column 2: position y (pc)\n')
            f.write('# column 3: position z (pc)\n')
            f.write('# column 4: smoothing length (pc)\n')
            f.write('# column 5: star formation rate (Msun/yr)\n')
            f.write('# column 6: metallicity (1)\n')
            f.write('# column 7: log10 of compactness (1)\n')
            f.write('# column 8: ISM Pressure (Pa)\n')
            f.write('# column 9: PDR Covering Factor (1)\n')
            
            log_compactness = 5 # Groves et al. 2008 (assumed constant outside of far infrared regime)
            ism_pressure = 1.38e-12 # Groves et al. 2008, converted log(P0/kB) = 5 to Pa (assumed constant outside of far infrared regime)
            pdr_covering_factor = 0.2 # Jonsson et al. 2010
            
            for i in range(len(x)):
                # MAPPINGS III Library SED model
                if age[i] < 1e7:
                    SFR = star_formation_rate(initial_mass[i]) # assumed SFR is constant in the last 10 Myr, i.e. Mi / 10 Myr.
                    f.write(f'{x[i]} {y[i]} {z[i]} {smoothing_length[i]} {SFR} {metallicity[i]} {log_compactness} {ism_pressure} {pdr_covering_factor}')
        
        
        
