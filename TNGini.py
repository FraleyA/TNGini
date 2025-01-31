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


def get_subhalo_position(subbox_dir, subhalo_id, snap_num):
    """ Return subhalo position in snapshot. """
    
    with h5py.File(subbox_dir, 'r') as hf:
        subhalo = np.where(hf['SubhaloIDs'][:] == subhalo_id)[0]
        subhalo_position = hf['SubhaloPos'][subhalo, snap_num, :]
        return subhalo_position[0]

    
def mass_density_in_pixels(path, snap_num, p_type, center, subbox_num=None, view='xy', box_height=5, box_length=20, box_width=20, Nbins=80, change_viewing_angle=False, theta=None, phi=None, align=False):
    """ Returns an image where pixel values represent stellar mass densities (unsorted, unprocessed).
        
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
         
    for idx in np.arange(Num):
        mass_density[int((axis1[idx] - 0.5*(-box_length))/h),int((axis2[idx] - 0.5*(-box_width))/h)] += prop[idx]
                
    return mass_density


def gini_plot(mass_density, style='seaborn-v0_8-muted', color1=2, color2=0, save_name=None, dpi=300):
    """ Just a plot of the Gini area between the galaxy distribution curve and L(p) = p.
        color1 and color2: Color based on the chosen matplotlib style parameter, and rcParams index.
    """

    # Apply mask to mass_density.
    galaxy_pixels = mass_density > 0
    mass_density = np.sort(mass_density[galaxy_pixels])
    mass_density.flatten()

    # Percentage of faintest pixels.
    pixels = np.arange(1, len(mass_density) + 1)
    percent_pixels = []
    for i in pixels:
        percent_pixels.append(i / len(pixels))

    percent_pixels = np.array(percent_pixels)

    # Percentage of total flux.
    percent_mass_density = []
    for i in range(1, len(mass_density) + 1):
        percent_mass_density.append(np.sum(mass_density[0:i]) / np.sum(mass_density))

    percent_mass_density = np.array(percent_mass_density)

    # Plot the processed mass_density data.
    plt.style.use(style)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color1 = colors[color1]
    color2= colors[color2]
    
    fig, ax = plt.subplots(dpi=dpi)
    ax.plot(pixels/len(pixels), pixels/len(pixels), color=color2, label=r'$L(p) = p$')
    ax.plot(percent_pixels, percent_mass_density, color=color1, label=r'$L(p)$')
    plt.fill_between(percent_pixels, pixels/len(pixels), percent_mass_density, color='grey', alpha=0.4)
    plt.text(0.55, 0.35, r'$G_{\rho}$', fontsize='xx-large')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(r'$p = \%\:pixels$')
    ax.set_ylabel(r'$L(p)$')
    ax.set_title('Gini Coefficient Mass Density')

    if save_name != None:
        fig.savefig(save_name)


def Gini(mass_density):
    """ Gini coefficient describes the distribution of light throughout a galaxy. This function
        calculates the G value of a merger, and is used to track G over the evolution of the merger.
        Takes an NxN array of pixels (image of galaxy).
    """
    
    # Only include galaxy pixels, sort the array
    galaxy_pixels = mass_density > 0
    ranked_mass_density = np.sort(mass_density[galaxy_pixels])
    
    # Increase pixel index by 1 for Gini because a 0th pixel does not make sense
    pixels = np.arange(0, len(ranked_mass_density))
    n = len(pixels)
    a = 1 / (np.mean(ranked_mass_density) * n * (n - 1)) # Coefficient outside of the summation
    
    # Initialize Gini coefficient
    summation = 0
        
    # Sum over each pixel to evaluate G
    for i in pixels:
        summation += (2 * (i + 1) - n - 1) * ranked_mass_density[i]
    
    return a * summation
        

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
                                                  box_height=box_height, box_length=box_length, box_width=box_width, Nbins=Nbins, 
                                                  change_viewing_angle=change_viewing_angle, theta=theta, phi=phi
                                                 )
            galaxy_pixels = mass_density > 0 # Only include pixels that contain stellar particles
            mass_density = np.sort(mass_density[galaxy_pixels]) # Sort pixels for gini calculation
            mass_density.flatten()
            
        else:
            mass_density = mass_density_in_pixels(path, snapshot, p_type, subhalo_center, subbox_num=subbox_num, view='xy', 
                                                  box_height=box_height, box_length=box_length, box_width=box_width, Nbins=Nbins, 
                                                 )
            galaxy_pixels = mass_density > 0 # Only include pixels that contain stellar particles
            mass_density = np.sort(mass_density[galaxy_pixels]) # Sort pixels for gini calculation
            mass_density.flatten()
            
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
            ap.galaxy2Dplots(path, i, '4', 'Density', subbox_num=1, view='xy', Nbins=80,
                             box_height=10, box_length=100, box_width=100, centre=subhalo_center, change_viewing_angle=True, theta=theta, phi=phi, vmin=-1.0, vmax=2.0, save_name=save_path+f'/merger_rotated_snap_{i}.png'
                            )
        
        return lookback_time, gini_coefficients
    
    # Save snapshots corresponding to the highlighted G values.
    snap_list = [snap_range[0], snap_range[0]+min_mask, snap_range[0]+max_mask, snap_range[-1]]
    for i in snap_list:
        ap.galaxy2Dplots(path, i, '4', 'Density', subbox_num=1, view='xy', Nbins=80,
                         box_height=10, box_length=100, box_width=100, centre=get_subhalo_position(subbox_dir, subhalo_id, i),
                         vmin=-1.0, vmax=2.0, save_name=save_path+f'/merger_snap_{i}.png'
                        )
        
    return lookback_time, gini_coefficients


def M_total_minimum(mass_density, maximum_iterations=1000, tolerance=1e-6, return_center=False):
    """ Takes the unprocessed merger snapshot, and uses gradient descent to pinpoint the minimum
        value of M_total.
        
        IMPORTANT: This function assumes that the mass_density array is centered on the subhalo position.
        Gradient descent might not land on the global minimum otherwise.
        
        mass_density: NxN pixel image of a galaxy merger.
        maximum_iterations: Iteration limit before the algorithm stops.
        return_center: returns (x_center, y_center) instead of the flux value at that location.
    """
    
    # Camera dimensions, i.e. the pixel coordinates (x, y)
    x_len = len(mass_density[0])
    y_len = len(mass_density[1])

    # Initial guess at the middle pixel because that is the subhalo center
    x_center = x_len / 2
    y_center = y_len / 2
    
    # Iteration counter
    iteration = 0

    # Index coordiantes of each pixel on the "camera"
    x_coords = np.arange(0, x_len)[:, None]
    y_coords = np.arange(0, y_len)[None, :]

    # Initial M_total
    M_total = np.sum(mass_density * ((x_coords - x_center)**2 + (y_coords - y_center)**2))

    # Need to implement feature that prevents oscillating between two positions around the minimum.
    visited_points = {} # {position; # of times visited}
    M_total_list = [M_total]

    while iteration < maximum_iterations:

        current_center = (x_center, y_center)

        # Count the number of visits; 0 initializes the value if it's the first entry
        visited_points[current_center] = visited_points.get(current_center, 0) + 1

        # M_total_new resets to 0 at the beginning of each iteration
        M_total_new = 0

        # M_total gradients with respect to x_center, y_center
        x_grad = -2 * np.sum(mass_density * (x_coords - x_center))
        y_grad = -2 * np.sum(mass_density * (y_coords - y_center))

        penalty_score = visited_points[current_center]

        # Dynamic learning rate: decreases inversely proportional to penalty score
        learning_rate = 1 / (1 + 0.2*penalty_score)

        # Negative sign ensures we step opposite of the gradient, i.e. gradient descent
        x_step = int(np.floor(-learning_rate * x_grad))
        y_step = int(np.floor(-learning_rate * y_grad))
        
        # Ensure the new position is between the first and last index
        if 0 <= (x_center + x_step) <= (x_len - 1):
            x_center = x_center + x_step
        if 0 <= (y_center + x_step) <= (y_len - 1):
            y_center = y_center + y_step

        # Compute M_total_new
        M_total_new = np.sum(mass_density * ((x_coords - x_center)**2 + (y_coords - y_center)**2))

        M_total_list.append(M_total_new)
        difference = M_total_new - M_total

        # Compare the old M_total to the new M_total
        if abs(difference) < tolerance:
            break

        # Reassign M_total and add iteration to counter
        M_total = M_total_new
        iteration += 1
        
        if return_center:
            return (x_center, y_center)
        
    return M_total


def M20_calc(mass_density, maximum_iterations=1000, tolerance=1e-6):
    """ The second-order moment of the brightest 20% of the mergering galaxies flux based on the 
        subhalo position. Definition and procedure for calculating M20 comes from Lotz, et. al (2004).
        
        mass_density: NxN pixel image of a galaxy merger.
        maximum_iterations: Iteration limit before the algorithm stops (M_total_minimum uses gradient descent). 
    """

    # Processed only including ranked galaxy pixels, now 1D-array; slicing reverses
    # the array such that the pixels are ordered in descending order (sum over brightest when iterated over)
    galaxy_pixels = mass_density > 0
    ranked_mass_density = np.sort(mass_density[galaxy_pixels])[::-1]

    # Store twenty percent of the total flux
    twenty_percent = 0.2 * np.sum(ranked_mass_density)

    # Iterate to find the pixels to include in 20% of galaxy flux
    threshold = 0
    pixels = np.array([])
    for pixel_flux in ranked_mass_density:
        threshold += pixel_flux
        if threshold < twenty_percent:
            pixels = np.append(pixels, pixel_flux)
        elif threshold > twenty_percent:
            break
    
    # Initialize M_i summation
    M_i = 0
    
    # Get the (x_center, y_center) coordinates
    x_center, y_center = M_total_minimum(mass_density, maximum_iterations=maximum_iterations, tolerance=tolerance, return_center=True)
    
    # Calculate M_i by summing over the brightest pixels
    for pixel_flux in pixels:
        x, y = np.where(mass_density == pixel_flux)
        M_i += pixel_flux * ((x - x_center)**2 + (y - y_center)**2)
    
    # Unprocessed image that determines the galaxy center pixel, and the normalization factor M_tot
    M_tot = M_total_minimum(mass_density, maximum_iterations=maximum_iterations, tolerance=tolerance)
    
    # This is what it was all for!
    M_20 = np.log10(M_i / M_tot)
    
    return M_20[0]


def G_vs_M20_plot(base_path, subbox_path, subhalo_id, snap_range, subbox_num=1, style='seaborn-v0_8-muted', figsize=(10, 6), dpi=300, cmap='inferno', save_path='/home/fraley.a/merger_morphology/plots/Gini_vs_M20/', save_name=None):
    """ Plot Gini coefficient vs. M20 at the respective snapshot. """
    
    lookback_time_vals = np.array([])
    G_vals = np.array([])
    M20_vals = np.array([])

    for snap in snap_range:

        # Get the scale factor for conversion to lookback time
        header = il.snapshot.loadHeader(base_path, snap, subbox_num)
        scale_factor = header.get('Time')

        # Set up the IllustrisTNG cosmological parameters to compute the lookback time
        cosmology = FlatLambdaCDM(H0=67.74 * units.km / units.s / units.Mpc,
                                  Om0=0.3089,
                                  Ob0=0.0486
                                 )

        # Convert scale factor array into redshift values
        redshift = (1 / scale_factor) - 1
        lookback_time = cosmology.lookback_time(redshift).to(units.Gyr).round(3).value
        lookback_time_vals = np.append(lookback_time_vals, lookback_time)

        subhalo_center = get_subhalo_position(subbox_path, subhalo_id, snap)
        
        # Stellar mass density, p_type = 4
        mass_density = mass_density_in_pixels(base_path, snap, '4', subhalo_center, subbox_num=subbox_num)

        G = Gini(mass_density)
        G_vals = np.append(G_vals, G)

        M20 = M20_calc(mass_density)
        M20_vals = np.append(M20_vals, M20)

    plt.style.use(style)
    colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.grid(True, zorder=0)

    # M20 at the respective lookback time
    im = ax.scatter(M20_vals, G_vals, c=lookback_time_vals, cmap=cmap, edgecolors='black', lw=0.5, s=25, zorder=2)
    cbar = fig.colorbar(im)
    cbar.set_label(r'Lookback Time [$Gyr$]', labelpad=25)
    ax.set_xlabel(r'$M_{20}$')
    ax.set_ylabel(r'$G_{\rho}$')
    plt.show()
    
    if save_name != None:
        fig.savefig(save_path + save_name)