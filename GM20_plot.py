import numpy as np
from TNGini import G_vs_M20_plot
import sys

# Merger tree directories: Alter the base_path as needed
base_path = '/orange/lblecha/IllustrisTNG/Runs/TNG100-1/output'

if __name__ == "__main__":

    if len(sys.argv) >= 7:
        initial_snap = int(sys.argv[1])
        final_snap = int(sys.argv[2])
        subhalo_id = int(sys.argv[3])
        subbox_num = int(sys.argv[4])
        subbox_path = sys.argv[5]
        save_name = sys.argv[6]
        
        if len(sys.argv) > 8:
            print('Too many command line args ({sys.argv}).')
            sys.exit()

    else:
        print('Expecting 7 command line args: initial_snap, final_snap, subhalo_id, subbox_num, subbox_path, save_name.')
        sys.exit()
    
    snap_range = np.arange(initial_snap, final_snap)
    G_vs_M20_plot(base_path, subbox_path, subhalo_id, snap_range, subbox_num=subbox_num, save_name=save_name)