import sys
import h5py
from ssl_mergers import merger_candidates, tree_traversal


def write_merger_histories(base_path, subboxNum=0, SSLSnapNum=99, save_dir='/home/fraley.a/merger_morphology/IllustrisTNG/Runs', TNG_run='TNG50-1'):
    
    # Get the list of candidate subhalos
    subfind_candidates = merger_candidates(base_path, subboxNum, SSLSnapNum)

    # Open an HDF5 file to store results
    output_file = TNG_run + f'_subbox{subboxNum}_subhalo_list_mergers.hdf5'
    with h5py.File(save_dir + f'/{TNG_run}/' + output_file, 'w') as hdf:
        for subhalo_id in subfind_candidates:
            print(f'Processing Subhalo ID: {subhalo_id}...')
            
            # Get the merger history for this subhalo
            data = tree_traversal(base_path, subhalo_id, subboxNum, SSLSnapNum)

            # Store the data in the HDF5 file
            subhalo_group = hdf.create_group(str(subhalo_id))
            for key, value in data[subhalo_id].items():
                subhalo_group.create_dataset(key, data=value)

    print(f'{TNG_run} subbox {subboxNum} merger histories saved to {output_file}.')
    

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        subbox_num = int(sys.argv[1])
        TNG_run = str(sys.argv[2])
        
        if len(sys.argv) > 3:
            print('Too many command line args ({sys.argv}).')
            sys.exit()

    else:
        print('Expecting 2 command line args: subbox_num and TNG_run.')
        sys.exit()
    
    # Merger tree directories: Alter the base_path as needed
    base_path = f'/orange/lblecha/IllustrisTNG/Runs/{TNG_run}/output'
    
    write_merger_histories(base_path, subboxNum=subbox_num, TNG_run=TNG_run)