import numpy as np
from TNGini import giniM20_plot
import sys

# Merger tree directories: Alter the base_path as needed
base_path = '/orange/lblecha/IllustrisTNG/Runs/TNG100-1/output'

if __name__ == "__main__":

    if len(sys.argv) >= 9:
        subbox_path = str(sys.argv[1])
        initial_snap = int(sys.argv[2])
        final_snap = int(sys.argv[3])
        subhalo_id = int(sys.argv[4])
        subbox_num = int(sys.argv[5])
        rotate = str(sys.argv[6])
        visualize = str(sys.argv[7])
        
        if rotate.lower() in ['true', 'yes', 'y', '1']:
            change_viewing_angle = True
        elif rotate.lower() in ['false', 'no', 'n', '0']:
            change_viewing_angle = False
            
        if visualize.lower() in ['true', 'yes', 'y', '1']:
            visualize_merger = True
        elif visualize.lower() in ['false', 'no', 'n', '0']:
            visualize_merger = False
        
        save_path = str(sys.argv[8])
        save_name = str(sys.argv[9])
        
        if len(sys.argv) > 10:
            print('Too many command line args ({sys.argv}).')
            sys.exit()

    else:
        print('Expecting 8 command line args: subbox_path, initial_snap, final_snap, subhalo_id, subbox_num, rotate, visualize, save_path, save_name.')
        sys.exit()
    
    snap_range = np.arange(initial_snap, final_snap)
    p_type = '4'
    giniM20_plot(base_path, subbox_path, subhalo_id, snap_range, p_type, subbox_num=subbox_num, 
                 change_viewing_angle=change_viewing_angle, visualize_merger=visualize_merger, save_path=save_path, save_name=save_name, save_data=True
                )