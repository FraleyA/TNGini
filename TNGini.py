import sys
sys.path.append('/home/fraley.a/packages') # Tell python where your packages live

import h5py
import numpy as np
import illustris_python as il
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as units
from arepo_python_tools.global_props import get_particle_data
import arepo_python_tools as ap


# Subbox directories
subbox0 = '/orange/lblecha/IllustrisTNG/Runs/TNG100-1/postprocessing/SubboxSubhaloList/subbox0_99.hdf5' # 7.5 cMpc/h
subbox1 = '/orange/lblecha/IllustrisTNG/Runs/TNG100-1/postprocessing/SubboxSubhaloList/subbox1_99.hdf5' # 7.5 cMpc/h

# Subbox centers from the TNG website
TNG100_subbox0_centerer = np.array([9000, 17000, 63000], dtype=int)
TNG100_subbox1_centerer = np.array([37000, 43500, 67500], dtype=int)

# Merger tree directories
base_path = '/orange/lblecha/IllustrisTNG/Runs/TNG100-1/output'


def get_subhalo_position(subbox_dir, subhalo_id, snap_num):
    """ Return subhalo position in snapshot. """
    
    with h5py.File(subbox_dir, 'r') as hf:
        subhalo = np.where(hf['SubhaloIDs'][:] == subhalo_id)[0]
        subhalo_position = hf['SubhaloPos'][subhalo, snap_num, :]
        return subhalo_position[0]

    
def mass_density_in_pixels(path, snap_num, p_type, center, subbox_num=None, view='xy', box_height=5, box_length=20, box_width=20, Nbins=80, change_viewing_angle=False, theta=None, phi=None, align=False):
    """ Returns the mass density of each galaxy pixel sorted in increasing order.
        
        subbox_num: 0, 1 or 2, left as optional so that the script may work with unmodified Illustris package in full box snapshots.
        Nbins: Kept at a low number to reproduce images closer to the resolution of actual observations.
        change_viewing_angle: Randomize the viewing angle which affects the gini values. 
        theta and phi: The values used to rotate the polar and azimuthal angles, respectively. 
        align: Align the viewing plane with the angular momentum axis. 
        
    """
    
    header = il.snapshot.loadHeader(path, snap_num, subbox_num)
    boxsize = header.get('BoxSize')
    
    # Load fields needed for stellar density calculation.
    load_fields = ['Masses','Coordinates']
        
    particle_data = get_particle_data(path, snap_num, p_type, load_fields, subbox_num)
    
    if align == False:
        if len(particle_data['Coordinates']) < 2:
            x_coord = np.array([])
            y_coord = np.array([])
            z_coord = np.array([])
        elif change_viewing_angle == True: # My addition to Aneesh's code to choose random viewing angles.
            if theta == None and phi == None:
                particle_positions = particle_data['Coordinates'] - center
            
                phi = np.random.uniform(0.0, 2.0*np.pi)
                cos_theta = np.random.uniform(-1.0, 1.0) # rule for uniform distribution of points on a sphere
    
                # Now, we can safely get theta from arccos(theta)
                theta = np.arccos(cos_theta)
    
                # Rotation matrix about x-axis in terms of the polar angle, theta.
                Rx = np.array([[1, 0, 0],
                              [0, np.cos(theta), -np.sin(theta)],
                              [0, np.sin(theta), np.cos(theta)]]
                             )
    
                # Rotation matrix about the z-axis in terms of the azimuthal angle, phi.
                Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                              [np.sin(phi), np.cos(phi), 0],
                              [0, 0, 1]]
                             )
        
                # total rotation matrix
                R = Rx @ Rz
    
                # Apply the rotation to the dataset
                new_positions = np.einsum('ij,kj->ki', R, particle_positions)
                x_coord = new_positions[:, 0]
                y_coord = new_positions[:, 1]
                z_coord = new_positions[:, 2]
            
            else:
                particle_positions = particle_data['Coordinates'] - center
                
                # Rotation matrix about x-axis in terms of the polar angle, theta.
                Rx = np.array([[1, 0, 0],
                              [0, np.cos(theta), -np.sin(theta)],
                              [0, np.sin(theta), np.cos(theta)]]
                             )
    
                # Rotation matrix about the z-axis in terms of the azimuthal angle, phi.
                Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                              [np.sin(phi), np.cos(phi), 0],
                              [0, 0, 1]]
                             )
            
                # total rotation matrix
                R = Rx @ Rz
    
                # Apply the rotation to the dataset
                new_positions = np.einsum('ij,kj->ki', R, particle_positions)
                x_coord = new_positions[:, 0]
                y_coord = new_positions[:, 1]
                z_coord = new_positions[:, 2]
        
        else:
            x_coord = particle_data['Coordinates'][:, 0] - center[0]
            y_coord = particle_data['Coordinates'][:, 1] - center[1]
            z_coord = particle_data['Coordinates'][:, 2] - center[2]
    
    else:
        angular_momentum_data = get_particle_data(path, snap_num, '045', ['Coordinates','Masses','Velocities'], subbox_num)
        pos = angular_momentum_data['Coordinates'] - center
        r = np.linalg.norm(pos,axis=1)
        pos = pos[r<5]
        masses = angular_momentum_data['Masses'][r<5]
        vel = angular_momentum_data['Velocities'][r<5]
        total_angular_momentum = np.array([np.sum(masses*(pos[:,1]*vel[:,2]-pos[:,2]*vel[:,1])),
                                           np.sum(masses*(pos[:,2]*vel[:,0]-pos[:,0]*vel[:,2])),
                                           np.sum(masses*(pos[:,0]*vel[:,1]-pos[:,1]*vel[:,0]))]
                                         )
        z_axis = total_angular_momentum / np.linalg.norm(total_angular_momentum)

        ct = z_axis[2]/np.sqrt(z_axis[0]**2+z_axis[1]**2+z_axis[2]**2)
        st = np.sqrt(1-ct**2)
        cp = z_axis[0]/np.sqrt(z_axis[0]**2+z_axis[1]**2)
        sp = z_axis[1]/np.sqrt(z_axis[0]**2+z_axis[1]**2)
        
        if len(particle_data['Coordinates']) < 2:
            x1 = np.array([])
            y1 = np.array([])
            z1 = np.array([])
        
        else:
            x1 = particle_data['Coordinates'][:,0] - center[0]
            y1 = particle_data['Coordinates'][:,1] - center[1]
            z1 = particle_data['Coordinates'][:,2] - center[2]

        x_coord = x1*ct*cp+y1*sp*ct-st*z1
        y_coord = -x1*sp+y1*cp
        z_coord = x1*st*cp+y1*st*sp+z1*ct

    if (view == 'xy'):
        axis1 = x_coord; axis2 = y_coord; axis3 = z_coord
    if (view == 'yz'):
        axis1 = y_coord; axis2 = z_coord; axis3 = x_coord
    if (view == 'xz'):
        axis1 = x_coord; axis2 = z_coord; axis3 = y_coord
    if (view == 'yx'):
        axis1 = y_coord; axis2 = x_coord; axis3 = z_coord
    if (view == 'zy'):
        axis1 = z_coord; axis2 = y_coord; axis3 = x_coord
    if (view == 'zx'):
        axis1 = z_coord; axis2 = x_coord; axis3 = y_coord
        
    mask1 = (axis3 > -box_height/2.0) & (axis3 < box_height/2.0) & (axis1>0.5*(-box_length))&(axis1<0.5*(box_length))&(axis2>0.5*(-box_width))&(axis2<0.5*(box_width))
    
    axis1 = axis1[mask1]
    axis2 = axis2[mask1]
    
    Num = len(axis1)

    ax1 = np.linspace(0.5*(-box_length),0.5*(box_length),Nbins)
    h = np.diff(ax1)[0]
    ax2 = np.arange(0.5*(-box_width),0.5*(box_width)+0.9*h,h)
    
    lx1 = len(ax1)
    lx2 = len(ax2)
    
    prop = particle_data['Masses'][mask1]
    
    # Binning method.
    mass_density = np.zeros((lx1, lx2))
    proj_freq = np.zeros((lx1, lx2))
         
    for idx in np.arange(Num):
        mass_density[int((axis1[idx] - 0.5*(-box_length))/h),int((axis2[idx] - 0.5*(-box_width))/h)] += prop[idx]
        proj_freq[int((axis1[idx] - 0.5*(-box_length))/h),int((axis2[idx] - 0.5*(-box_width))/h)] += 1
    
    galaxy_pixels = mass_density > 0 # Only include pixels that contain particles
    mass_density = np.sort(mass_density[galaxy_pixels])
    mass_density = mass_density.flatten()
                
    return mass_density


def gini_over_time(path, subbox_dir, subhalo_id, snap_range, p_type, save_path, subbox_num=None, view='xy', 
                   box_height=5, box_length=20, box_width=20, Nbins=80, 
                   align=False, change_viewing_angle=False, theta=None, phi=None):
    
    """ Returns an array of gini coefficients at each respective lookback time. """
    
    # Initialize list of G values at a given scale factor/lookback time.
    gini_coefficients = []
    scale_factor = []
    
    for snapshot in snap_range:
        header = il.snapshot.loadHeader(path, snapshot, 1)
        time = header.get('Time')
        scale_factor.append(time)
        
        subhalo_center = get_subhalo_position(subbox_dir, subhalo_id, snapshot)
        
        if change_viewing_angle:
            mass_density = mass_density_in_pixels(path, snapshot, p_type, subhalo_center, subbox_num=subbox_num, view='xy', 
                                                  box_height=10, box_length=box_length, box_width=box_width, Nbins=Nbins, 
                                                  change_viewing_angle=change_viewing_angle, theta=theta, phi=phi
                                                 )
        else:
            mass_density = mass_density_in_pixels(path, snapshot, p_type, subhalo_center, subbox_num=subbox_num, view='xy', 
                                                  box_height=10, box_length=box_length, box_width=box_width, Nbins=Nbins, 
                                                 )
            
        pixels = np.arange(1, len(mass_density) + 1)
        n = len(pixels)
        a = 1 / (np.mean(mass_density) * n * (n - 1)) # coefficient outside of the summation
        result = 0 # initialize Gini
        
        # Sum over each pixel to evaluate G.
        for i in range(0, len(pixels)):
            result += (2 * (i + 1) - n - 1) * mass_density[i]
    
        gini_coefficients.append(a * result)
    
    scale_factor = np.array(scale_factor)
    
    # Set up the IllustrisTNG cosmological parameters to compute the lookback time.
    cosmology = FlatLambdaCDM(H0=67.74 * units.km / units.s / units.Mpc,
                              Om0=0.3089,
                              Ob0=0.0486
                             )

    # Convert scale factor array into redshift values.
    redshift = (1 / scale_factor) - 1
    lookback_time = cosmology.lookback_time(redshift).to(units.Gyr).round(3).value
    print(lookback_time)
    
    gini_coefficients = np.array(gini_coefficients)
    print(gini_coefficients)
    
    plt.style.use('seaborn-v0_8-muted')
    colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    fig, ax = plt.subplots(dpi=300)
    
    ax.grid(True, zorder=0)
    
    # All of the gini coefficients.
    ax.scatter(lookback_time, gini_coefficients, c=colors[0], edgecolors='black', lw=0.5, s=25, zorder=2)
    
    if change_viewing_angle:
        #save_path = '/home/fraley.a/merger_tree_project/subbox1_mergers/MergerRotatedTest/'
        
        # Choose points throughout the merger to highlight for comparison at random viewing angles.
        image_times = np.array([lookback_time[0], lookback_time[round(len(lookback_time) / 4)], lookback_time[round(len(lookback_time) / 2)], lookback_time[round(3 * len(lookback_time) / 4)], lookback_time[-1]])
        image_gini_coefficients = np.array([gini_coefficients[0], gini_coefficients[round(len(gini_coefficients) / 4)], gini_coefficients[round(len(gini_coefficients) / 2)], gini_coefficients[round(3 * len(gini_coefficients) / 4)], gini_coefficients[-1]])
        ax.scatter(image_times, image_gini_coefficients, c=colors[2], edgecolors='black', lw=0.5, s=30, zorder=3)
        
        # Note rotations for the polar and azimuthal angles visually on the plot.
        ax.text(0.85, 0.90, s=r'$\theta$ = ' + f'{round(theta * (180 / np.pi))}' + r'$^{\circ}$', transform=ax.transAxes)
        ax.text(0.85, 0.85, s=r'$\phi$ = ' + f'{round(phi * (180 / np.pi))}' + r'$^{\circ}$', transform=ax.transAxes)
    
    else:
        #save_path = '/home/fraley.a/merger_tree_project/subbox1_mergers/'
        
        # Highlight the first and last G, minimum G, and maximum G.
        min_mask = np.where(gini_coefficients == np.min(gini_coefficients))[0][0]
        max_mask = np.where(gini_coefficients == np.max(gini_coefficients))[0][0]
        image_times = np.array([lookback_time[0], lookback_time[min_mask], lookback_time[max_mask], lookback_time[-1]])
        image_gini_coefficients = np.array([gini_coefficients[0], gini_coefficients[min_mask], gini_coefficients[max_mask], gini_coefficients[-1]])
        ax.scatter(image_times, image_gini_coefficients, c=colors[2], edgecolors='black', lw=0.5, s=30, zorder=3)
    
    ax.set_xlabel(r'Lookback Time [$Gyr$]')
    ax.invert_xaxis() # Beginning of merger at the origin.
    ax.set_ylabel(r'$G_{\rho}$')
    
    # Save the figure to the desired directory.
    fig.savefig(save_path + '/gini_vs_time.png')
    
    # Make sure theta and phi are randomly generated outside of the function in this case.
    if change_viewing_angle:
        if theta == None and phi == None:
            print('Error: Choose theta and phi.')
            return
        
        snap_list = [snap_range[0], snap_range[round(len(snap_range) / 4)], snap_range[round(len(snap_range) / 2)], snap_range[round(3 * len(snap_range) / 4)], snap_range[-1]]
        
        for i in snap_list:
            subhalo_center = get_subhalo_position(subbox_dir, subhalo_id, i)
            ap.galaxy2Dplots(base_path, i, '4', 'Density', subbox_num=1, view='xy', Nbins=80,
                             box_height=10, box_length=100, box_width=100, centre=subhalo_center, change_viewing_angle=True, theta=theta, phi=phi, vmin=-1.0, vmax=2.0, save_name=save_path+f'/merger_rotated_snap_{i}.png'
                            )
        
        return lookback_time, gini_coefficients
    
    # Save snapshots corresponding to the highlighted G values.
    snap_list = [snap_range[0], snap_range[0]+min_mask, snap_range[0]+max_mask, snap_range[-1]]
    for i in snap_list:
        ap.galaxy2Dplots(base_path, i, '4', 'Density', subbox_num=1, view='xy', Nbins=80,
                         box_height=10, box_length=100, box_width=100, centre=get_subhalo_position(subbox_dir, subhalo_id, i),
                         vmin=-1.0, vmax=2.0, save_name=save_path+f'/merger_snap_{i}.png'
                        )
        
    return lookback_time, gini_coefficients


if __name__ == "__main__":

    if len(sys.argv) >= 6:
        initial_snap = int(sys.argv[1])
        final_snap = int(sys.argv[2])
        subhalo_id = int(sys.argv[3])
        subbox_num = int(sys.argv[4])
        
        # Interpret the boolean argument
        rotate = sys.argv[5].lower()  # Convert to lowercase for case-insensitive matching
        if rotate in ('true', '1', 'yes', 'y'):
            change_viewing_angle = True
        elif rotate in ('false', '0', 'no', 'n'):
            change_viewing_angle = False
        else:
            print(f'Invalid boolean value for change_viewing_angle: {sys.argv[4]}')
            sys.exit(1)
            
        save_path = str(sys.argv[6])
        
        if len(sys.argv) > 7:
            print('Too many command line args ({sys.argv}).')
            sys.exit()

    else:
        print('Expecting 3 command line args: initial_snap, final_snap, subhalo_id, change_viewing_angle.')
        sys.exit()
    print(f'Display sys.argv params: {initial_snap}, {final_snap}, {subhalo_id}, {subbox_num}, {change_viewing_angle}, {save_path}')
    
    # Define snap range to iterate over for gini and plotting.
    snap_range = np.arange(initial_snap, final_snap + 1)
    
    # Rotate the plot if desired.
    if rotate:
        cos_theta = np.random.uniform(-1, 1)
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0, 2 * np.pi)
        gini_over_time(base_path, subbox1, subhalo_id, snap_range, '4', save_path, subbox_num=subbox_num, change_viewing_angle=change_viewing_angle, theta=theta, phi=phi)
    else:
        gini_over_time(base_path, subbox1, subhalo_id, snap_range, '4', save_path, subbox_num=subbox_num)