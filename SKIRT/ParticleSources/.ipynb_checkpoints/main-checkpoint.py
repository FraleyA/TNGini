import sys
sys.path.append('/home/fraley.a/packages')

from stellar_sources import stellar_data

base_path = '/orange/lblecha/fraley.a/IllustrisTNG/Runs/TNG50-1/output'

if __name__ == '__main__':
    
    if len(sys.argv) >= 4:
        subbox_snap_num = int(sys.argv[1])
        subbox_num = int(sys.argv[2])
        subfind_id = int(sys.argv[3])
        
    else:
        print('Expecting 3 command line arguements: subbox_snap_num, subbox_num, and subfind_id.')
        sys.exit()
    
    save_path = f'/orange/lblecha/fraley.a/SKIRT/Runs/TNG50-1/subbox{subbox_num}/subhalo{subfind_id}'
    stellar_data(base_path, subbox_snap_num, subbox_num, subfind_id, SED='Bruzual&Charlot', savePath=save_path)
    stellar_data(base_path, subbox_snap_num, subbox_num, subfind_id, SED='MAPPINGS', savePath=save_path)