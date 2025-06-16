import sys
sys.path.append('/home/fraley.a/packages')

import arepo_python_tools as ap
from TNGini import get_subhalo_position

if __name__ == '__main__':
    base_path = str(sys.argv[1])
    subboxNum = int(sys.argv[2])
    SubboxSubhaloListSnapNum = int(sys.argv[3])
    subhalo_id = int(sys.argv[4])
    
    subhalo_position = get_subhalo_position(base_path+'/postprocessing/SubboxSubhaloList/'+f'subbox{subboxNum}_{SubboxSubhaloListSnapNum}.hdf5', subhalo_id, SubboxSubhaloListSnapNum)
    p_type = '4' # stars
    particle_property = 'Density'
    save_name = f'/home/fraley.a/merger_morphology/TNG50_1_mergerSH{subhalo_id}.png'
    
    # subbox_num=None by default if not plotting the stellar density of a subbox snapshot
    ap.galaxy2Dplots(path=base_path, snap_num=SubboxSubhaloListSnapNum, p_type=p_type, particle_property=particle_property, subbox_num=None,
                     view='xy', box_height=5, box_length=60, box_width=60, Nbins=100, method='binning',
                     smooth=False, change_viewing_angle=False, theta=None, phi=None, align=False, centre=subhalo_position, save_name=save_name
                    )