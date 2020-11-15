import glob

for in_path in glob.glob("/home/yaelsc/Data/AirMSPI/clouds/cloud*.txt"):
    fin = open(in_path,"rt")
    out_path = in_path.replace("/home/yaelsc/Data/AirMSPI/clouds/",
                               "/home/yaelsc/Data/AirMSPI/clouds_airmspi/")
    fout = open(out_path,"wt")

    for line in fin:
        fout.write(line.replace(
            '0.050 0.050 0.000 0.040 0.080 0.120 0.160 0.200 0.240 0.280 0.320 0.360 0.400 0.440 0.480 0.520 0.560 0.600 0.640 0.680 0.720 0.760 0.800 0.840 0.880 0.920 0.960 1.000 1.040 1.080 1.120 1.160 1.200 1.240',
            '0.088 0.118 0.800 0.822 0.844 0.866 0.888 0.909 0.931 0.953 0.975 0.997 1.019 1.041 1.062 1.084 1.106 1.128 1.150 1.172 1.194 1.216 1.237 1.259 1.281 1.303 1.325 1.347 1.369 1.391 1.412 1.434 1.456 1.478'))

        # close input and output files
    fin.close()
    fout.close()