#!/bin/bash

files=(
#'/home/shubi/PycharmProjects/pyshdom/synthetic_cloud_fields/jpl_les/rico32x37x26.txt'
'/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped/BOMEX_21720_30x36x18_o2.txt'
'/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped/BOMEX_36000_42x41x26_820024.txt'
'/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped/BOMEX_50400_42x34x39_002485.txt'
#'/wdata/clouds_from_weizmann/CASS_50m_256x256x139_600CCN/cropped/CASS_28800_56x67x45_478420.txt'
#'/wdata/clouds_from_weizmann/CASS_50m_256x256x139_600CCN/cropped/CASS_30600_101x77x49_418131.txt'
)
for file in "${files[@]}"
do
  #echo "running script with $file"
  python CloudCT_simulate.py --cloudFieldFile $file
done