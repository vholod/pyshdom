#!/bin/bash

files=(
#'/home/shubi/PycharmProjects/pyshdom/synthetic_cloud_fields/jpl_les/rico32x37x26.txt'
#'/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped/BOMEX_21720_30x36x18_o2.txt'
#'/wdata/clouds_from_weizmann/CASS_50m_256x256x139_600CCN/cropped/CASS_18000_33x34x20_0411.txt'
'/wdata/clouds_from_weizmann/CASS_50m_256x256x139_600CCN/cropped/CASS_37800_25x21x23_8050.txt'
'/wdata/clouds_from_weizmann/CASS_50m_256x256x139_600CCN/cropped/CASS_28800_34x31x26_3042.txt'

)
for file in "${files[@]}"
do
  #echo "running script with $file"
  python CloudCT_simulate.py --cloudFieldFile $file
done