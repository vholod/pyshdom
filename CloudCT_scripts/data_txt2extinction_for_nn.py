import glob
import os

import scipy.io as sio

import shdom
from CloudCT_scripts.CloudCT_simulate_for_nn import load_run_params


def exists(p, msg):
    assert os.path.exists(p), msg


"""
save an extinction with 1/km units.
take inputs:
scat file
txt file describing the lwc and reff

"""


def main():
    # mie
    scat_file_path = "/home/yaelsc/PycharmProjects/pyshdom/mie_tables/polydisperse/Water_672nm.scat"
    exists(scat_file_path, 'scat_file_path not found!')
    # Mie scattering for water droplets
    mie = shdom.MiePolydisperse()
    print("reading the scat table from: {}".format(scat_file_path))
    mie.read_table(scat_file_path)

    clouds_txt_path = "/wdata/yaelsc/Data/fab_clouds_fields/clouds"
    clouds_mat_path = "/wdata/yaelsc/Data/fab_clouds_fields/betas"
    for cloud_txt_path in glob.glob(os.path.join(clouds_txt_path, "*.txt")):
        cloud_index = cloud_txt_path.split(clouds_txt_path)[1].replace('.txt', '').replace('/cloud', '')
        print(f'processing cloud {cloud_index}')
        cloud_mat_path = os.path.join(clouds_mat_path, f'gt_cloud{cloud_index}.mat')
        exists(cloud_txt_path, 'csv_txt not found!')
        if os.path.exists(cloud_mat_path):
            print(f'cloud {cloud_index} already exists! skipping')
            continue

        # Generate a Microphysical medium
        droplets = shdom.MicrophysicalScatterer()
        droplets.load_from_csv(cloud_txt_path, veff=0.1)

        # threshold
        run_params = load_run_params(params_path="run_params_cloud_ct_nn_rico.yaml")
        mie_options = run_params['mie_options']
        droplets.reff.data[droplets.reff.data <= mie_options['start_reff']] = mie_options['start_reff']
        droplets.reff.data[droplets.reff.data >= mie_options['end_reff']] = mie_options['end_reff']
        if len(droplets.veff.data[droplets.veff.data <= mie_options['start_veff']]) > 0:
            droplets.veff.data[droplets.veff.data <= mie_options['start_veff']] = mie_options['start_veff']
        if len(droplets.veff.data[droplets.veff.data >= mie_options['end_veff']]) > 0:
            droplets.veff.data[droplets.veff.data >= mie_options['end_veff']] = mie_options['end_veff']

        droplets.add_mie(mie)

        # extract the extinction:
        extinction = mie.get_extinction(droplets.lwc, droplets.reff, droplets.veff)
        extinction_data = extinction.data  # 1/km

        # save extintcion as mat file:
        sio.savemat(cloud_mat_path, dict(beta=extinction_data, lwc=droplets.lwc.data, reff=droplets.reff.data, veff=droplets.veff.data))
        print("saving the .mat file to: {}".format(cloud_mat_path))
        print('finished')

        """
        summary:
        The mat file must include the fields (beta,lwc,reff,veff)
    """


if __name__ == '__main__':
    main()
