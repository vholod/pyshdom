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
    scat_file_path = "/home/yaelsc/PycharmProjects/pyshdom_new/mie_tables/polydisperse/Water_672nm.scat"
    exists(scat_file_path, 'scat_file_path not found!')
    # Mie scattering for water droplets
    mie = shdom.MiePolydisperse()
    print("reading the scat table from: {}".format(scat_file_path))
    mie.read_table(scat_file_path)

    csv_path = "/home/yaelsc/PycharmProjects/DataTransdormationToBeta/bomexDataToMat/BOMEX_256x256x100_5000CCN_50m_micro_256/SHDOM_files_25_06_20"
    out_path = "/home/yaelsc/PycharmProjects/pyshdom_new/CloudCT_nn/Data/lwcs"
    for csv_txt in glob.glob(os.path.join(csv_path, "*.txt")):
        cloud_index = csv_txt.split(csv_path)[1].replace('.txt', '').replace('/cloud', '')
        out_mat = os.path.join(out_path, f'cloud{cloud_index}.mat')
        exists(csv_txt, 'csv_txt not found!')

        # Generate a Microphysical medium
        droplets = shdom.MicrophysicalScatterer()
        droplets.load_from_csv(csv_txt, veff=0.1)

        # threshold
        run_params = load_run_params(params_path="run_params_cloud_ct_nn_test.yaml")
        mie_options = run_params['mie_options']
        droplets.reff.data[droplets.reff.data >= mie_options['start_reff']] = mie_options['start_reff']
        droplets.reff.data[droplets.reff.data <= mie_options['end_reff']] = mie_options['end_reff']
        droplets.veff.data[droplets.veff.data >= mie_options['start_veff']] = mie_options['start_veff']
        droplets.veff.data[droplets.veff.data >= mie_options['end_veff']] = mie_options['end_veff']

        droplets.add_mie(mie)

        # extract the extinction:
        extinction = mie.get_extinction(droplets.lwc, droplets.reff, droplets.veff)
        extinction_data = extinction.data  # 1/km

        # save extintcion as mat file:
        sio.savemat(out_mat, dict(beta=extinction_data, lwc=droplets.lwc.data, reff=droplets.reff.data, veff=droplets.veff.data))
        print("saving the .mat file to: {}".format(out_mat))
        print('finished')

        """
        summary:
        The mat file must include the fields (beta,lwc,reff,veff)
    """


if __name__ == '__main__':
    main()
