import numpy as np
from TNGini import gini_evolution
import sys

# Subbox directories: Alter these as needed
subbox0 = '/orange/lblecha/IllustrisTNG/Runs/TNG100-1/postprocessing/SubboxSubhaloList/subbox0_99.hdf5' # 7.5 cMpc/h
subbox1 = '/orange/lblecha/IllustrisTNG/Runs/TNG100-1/postprocessing/SubboxSubhaloList/subbox1_99.hdf5' # 7.5 cMpc/h

# Subbox centers from the TNG website
TNG100_subbox0_center = np.array([9000, 17000, 63000], dtype=int)
TNG100_subbox1_center = np.array([37000, 43500, 67500], dtype=int)

# Merger tree directories: Alter base_path as needed
base_path = '/orange/lblecha/IllustrisTNG/Runs/TNG100-1/output'

if __name__ == "__main__":

    if len(sys.argv) >= 7:
        initial_snap = int(sys.argv[1])
        final_snap = int(sys.argv[2])
        subhalo_id = int(sys.argv[3])
        subbox_num = int(sys.argv[4])
        
        # Interpret the boolean argument for sys.argv
        rotate = sys.argv[5].lower()  # Convert to lowercase for case-insensitive matching
        if rotate in ('true', '1', 'yes', 'y'):
            change_viewing_angle = True
        elif rotate in ('false', '0', 'no', 'n'):
            change_viewing_angle = False
        else:
            print(f'Invalid boolean value for change_viewing_angle: {sys.argv[5]}')
            sys.exit()
            
        save_path = str(sys.argv[6])
        save_name = str(sys.argv[7])
        
        if len(sys.argv) > 8:
            print('Too many command line args ({sys.argv}).')
            sys.exit()

    else:
        print('Expecting 7 command line args: initial_snap, final_snap, subhalo_id, subbox_num, rotate, save_path, save_name.')
        sys.exit()
    
    snap_range = np.arange(initial_snap, final_snap + 1)
    p_type = '4' # stellar particles
    
    # Randomly rotate the plot if desired
    if rotate:
        gini_evolution(base_path, subbox1, subhalo_id, snap_range, p_type, subbox_num=subbox_num, change_viewing_angle=change_viewing_angle, save_path=save_path, save_name=save_name)
        
    # Default viewing plane
    else:
        gini_evolution(base_path, subbox1, subhalo_id, snap_range, p_type, subbox_num=subbox_num, save_path=save_path, save_name=save_name)
