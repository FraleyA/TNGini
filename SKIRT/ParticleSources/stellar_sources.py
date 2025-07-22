import sys
sys.path.append('/home/fraley.a/packages')

import os
import glob

import h5py

import numpy as np
from scipy.spatial import cKDTree

import astropy.units as units
from astropy.cosmology import FlatLambdaCDM


def snap_chunks(base_path, snap_num, subbox_num):
    """ Returns a sorted list of snapshot chunks.
        Example: [snap_subbox2_2783.0.hdf5, snap_subbox2_2783.1.hdf5, snap_subbox2_2783.10.hdf5, ...]  
    """
    
    snap_path = base_path + f'/subbox{subbox_num}_snapdirs/snap_subbox{subbox_num}_{snap_num}'
    snap_file = f'snap_subbox{subbox_num}_{snap_num}.*.hdf5'
    snap_pattern = os.path.join(snap_path, snap_file)
    chunk_list = sorted(glob.glob(snap_pattern))
    
    return chunk_list


def get_subhalo_position(subbox_path, subhalo_id, snap_num):
    """ Return the comoving position of the subhalo in the given snapshot per h [ckpc/h]. """
    
    with h5py.File(subbox_path, 'r') as hf:
        subhalo = np.where(hf['SubhaloIDs'][:] == subhalo_id)[0][0]
        subhalo_position = hf['SubhaloPos'][subhalo, snap_num, :]
        return subhalo_position

    
def get_scale_factor(subbox_path, snap_num):
    """ Return the scale factor of a given subbox snapshot. """
    
    with h5py.File(subbox_path, 'r') as hf:
        return hf['SubboxScaleFac'][snap_num]

    
def get_stellar_age(subbox_path, snap_num, GFM_StellarFormationTime, H0=67.74, Om0=0.3089, Ob0=0.0486):
    """ Use astropy FlatLambdaCDM with TNG cosmological parameters to obtain stellar formation time.
        GFM_StellarFormationTime: The scale factor corresponding to the when the stellar particle is born.
        Om0: Density of non-relativistic matter in units of critical density at z=0.
        Ob0: Density of baryonic in units of critical density at z=0.
    """
    
    cosmo = FlatLambdaCDM(H0=H0*units.km/units.s/units.Mpc, Om0=Om0, Ob0=Ob0)

    # Age of the universe in the specified snapshot.
    a = get_scale_factor(subbox_path, snap_num)
    z = (1 / a) - 1
    t = cosmo.age(z).to(units.yr).value

    # Age of the universe at the time the stellar population formed.
    z0 = (1 / GFM_StellarFormationTime) - 1
    t0 = cosmo.age(z0).to(units.yr).value

    # The time difference between the age of the universe in snapshot and the stellar formation time.
    return np.abs(t - t0) # units: [yr]


def get_smoothing_length(positions, tol=800):
    """ StellarHsml is unfortunately not present in the subbox snapshot data. 
        Use scipy.spatial.cKDTree for a quick nearest neighbor search.
        positions: (N, 3) where there are N stellar particles with coordinates (x, y, z).
        tol: Put a cap on the maximum smoothing length, i.e. 800 pc.
    """
    
    # Construct the cKDTree.
    tree = cKDTree(positions)
    
    # Query for the 32nd nearest neighbor.
    k = 32 + 1
    distances, _ = tree.query(positions, k=k)
    
    # Define StellarHsml as the 32nd nearest neighbor for each particle.
    StellarHsml = distances[:, 32]
    StellarHsml = np.minimum(StellarHsml, tol) # cap at tol [pc]
    
    return StellarHsml


def write_header(file_name, SED, snap_num, id, M_curr=False):
    """ Write file header for relevant stellar populations based on SED model.
        SED: 'Bruzual&Charlot' or 'MAPPINGS' 
        M_curr: Choose whether or not to write a current mass column.
    """
    
    if SED == 'Bruzual&Charlot':
        with open(file_name, 'w') as f:

            # Write the file header.
            f.write(f'# My Stellar Sources File Written From Python - Merger in Subbox Snapshot {snap_num} Related to SubfindID {id} from the Subbox Subhalo List\n')
            f.write('# column 1: position x (pc)\n')
            f.write('# column 2: position y (pc)\n')
            f.write('# column 3: position z (pc)\n')
            f.write('# column 4: smoothing length (pc)\n')
            f.write('# column 5: initial mass (Msun)\n')
            f.write('# column 6: metallicity (1)\n')
            f.write('# column 7: age (yr)\n')
            
    elif SED == 'MAPPINGS':
        with open(file_name, 'w') as f:
            
            # Write the file header.
            f.write(f'# My Stellar Sources File Written From Python - Merger in Subbox Snapshot {snap_num} Related to SubfindID {id} from the Subbox Subhalo List\n')
            f.write('# column 1: position x (pc)\n')
            f.write('# column 2: position y (pc)\n')
            f.write('# column 3: position z (pc)\n')
            f.write('# column 4: smoothing length (pc)\n')
            f.write('# column 5: star formation rate (Msun/yr)\n')
            f.write('# column 6: metallicity (1)\n')
            f.write('# column 7: log10 of compactness (1)\n')
            f.write('# column 8: ISM Pressure (Pa)\n')
            f.write('# column 9: PDR Covering Factor (1)\n')
            
    else:
        print('Specify the SED model to use: Bruzual&Charlot or MAPPINGS.')
        return
        
    
def write_file(file_name, x_pos, y_pos, z_pos, smoothing_lengths, SED, initial_masses=None, metallicities=None, ages=None, star_formation_rates=None):
    """ Iterate through stellar particle data to write it in line by line.
        x, y, z, h: array-like, units in [pc] positions and smoothing length. 
        SED: string, Bruzual&Charlot or MAPPINGS.
        Minit, Z, t, SFR: array-like, optional values depending on the SED model.
    """
    
    if SED == 'Bruzual&Charlot':
        with open(file_name, 'a') as f:

            # Write in the data.
            for x, y, z, h, Minit, Z, t in zip(x_pos, y_pos, z_pos, smoothing_lengths, initial_masses, metallicities, ages):
                f.write(f'{x} {y} {z} {h} {Minit} {Z} {t}\n')
                
    elif SED == 'MAPPINGS':  
        with open(file_name, 'a') as f:

            # Write data to the text file.
            for x, y, z, h, SFR, Z in zip(x_pos, y_pos, z_pos, smoothing_lengths, star_formation_rates, metallicities):

                # log(C), ISM Pressure, and PDR Covering Fraction are assumed constant based on Groves et al. (2008), Jonsson et al. (2010), and Rodriguez-Gomez et al. (2019).
                f.write(f'{x} {y} {z} {h} {SFR} {Z} {5} {1.38e-12} {0.2}\n')
                
    else:
        print('Specify a valid SED model: Bruzual&Charlot or MAPPINGS.')
        return

    
def stellar_data(base_path, snap_num, subbox_num, id, SED='Bruzual&Charlot', SSLSnapNum=99, maxDistance=200, maxSmoothingLength=800, hubble=0.6774, savePath='/orange/lblecha/fraley.a'):
    """ Create stellar data files for use in the SKIRT radiative transfer Monte Carlo simulation. 
        SED: 'BruzualCharlot' or 'MAPPINGS' depending on the value stellar_ages[i] at index, i.
        id: SubhaloIDs id from the subbox subhalo list (i.e. SubfindID).
        maxDistance [kpc]: maximum distance a particle can be from the subhalo's position.
        maxSmoothingLength [pc]: Cap on the smoothing length to prevent over-smoothing.
        NOTE: relative_position is the vector from the subhalo center to the star particle, I am choosing
        the subhalo center to be the origin of the simulated image.
    """
    
    # Initialize the file by writing a header.
    if SED == 'Bruzual&Charlot':
        file = savePath + '/older_pop.txt'
        write_header(file, SED, snap_num, id)
        
    elif SED == 'MAPPINGS':
        file = savePath + '/younger_pop.txt'
        write_header(file, SED, snap_num, id)
        
    else:
        print('Specify a valid SED model: Bruzual&Charlot or MAPPINGS.')
        return
    
    ### Extract all of the required data from TNG, and compute necessary fields. ###
    
    # Subbox subhalo list file path.
    subbox_path = base_path + f'/postprocessing/SubboxSubhaloList/subbox{subbox_num}_{SSLSnapNum}.hdf5'
    
    # Get the snapshot scale factor.
    a = get_scale_factor(subbox_path, snap_num)
    
    # Units: [kpc]
    subhalo_position = get_subhalo_position(subbox_path, id, snap_num) * (a / hubble)
    
    # List of .hdf5 files sorted by chunk.
    chunk_files = snap_chunks(base_path, snap_num, subbox_num)
    
    for chunk in chunk_files:
        
        f = h5py.File(chunk, 'r')
        star_particles = f['PartType4']
        
        # Units: [kpc]
        particle_positions = star_particles['Coordinates'] * (a / hubble)
        relative_positions = particle_positions - subhalo_position
        
        # Compare the distance of stellar particles to the subhalo position, and mask using maxDistance.
        relative_distances = np.linalg.norm(relative_positions, axis=1)
        mask = relative_distances < maxDistance
        
        # Only including particles within maxDistance of the subhalo position, units: [pc].
        relative_positions = relative_positions[mask] * 1e3
        
        # Calculate the 32nd nearest neighbor of each particle, i.e. h (smoothing length), units : [pc].
        # IMPORTANT: Calculate before applying age mask as to not exclude neighbors with a different SED model.
        smoothing_lengths = get_smoothing_length(relative_positions, tol=maxSmoothingLength)
        
        # Units: [Msun]
        initial_masses = star_particles['GFM_InitialMass'][mask] * (1e10 / hubble)
        current_masses = star_particles['Masses'][mask] * (1e10 / hubble)
        
        # Units: dimensionless (1)
        metallicities = star_particles['GFM_Metallicity'][mask]
        
        # Units: dimensionless (1), scale factors corresponding to birth for each pop.
        a0_vals = star_particles['GFM_StellarFormationTime'][mask]
        
        # Units: [yr]
        ages = get_stellar_age(subbox_path, snap_num, a0_vals)
        
        if SED == 'Bruzual&Charlot':
        
            # Stellar populations older than 10 Myr.
            bc_ages = ages > 1e7

            ### Apply age mask. ###
            relative_positions = relative_positions[bc_ages]

            # Position and smoothing length data.
            x_pos = relative_positions[:, 0]
            y_pos = relative_positions[:, 1]
            z_pos = relative_positions[:, 2]
            smoothing_lengths = smoothing_lengths[bc_ages]
            
            # SED Parameters.
            initial_masses = initial_masses[bc_ages]
            metallicities = metallicities[bc_ages]
            ages = ages[bc_ages]

            # Write the data to a file.
            write_file(file, x_pos, y_pos, z_pos, smoothing_lengths, 
                       SED, initial_masses=initial_masses, metallicities=metallicities, ages=ages
                      )
        
        elif SED == 'MAPPINGS':
        
            # Stellar populations younger than 10 Myr.
            map_ages = ages < 1e7

            ### Apply age mask. ###
            relative_positions = relative_positions[map_ages]

            # Position and smoothing length data.
            x_pos = relative_positions[:, 0]
            y_pos = relative_positions[:, 1]
            z_pos = relative_positions[:, 2]
            smoothing_lengths = smoothing_lengths[map_ages]

            # SED Parameters, SFR assumed constant over the last 10 Myr.
            star_formation_rates = current_masses[map_ages] / 1e7
            metallicities = metallicities[map_ages]
            ages = ages[map_ages]

            # Write the data to a file.
            write_file(file, x_pos, y_pos, z_pos, smoothing_lengths,
                       SED, metallicities=metallicities, star_formation_rates=star_formation_rates
                      )
        
        f.close()