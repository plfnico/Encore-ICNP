import numpy as np
import pandas as pd
import os
import sys
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def run_baseline_parallel(config):
    cmdLine = './ns3 run \"scratch/dumbbell {config}\"'.format(config=config)
    print(cmdLine)
    os.system(cmdLine)


if __name__ == "__main__":
    os.chdir(sys.path[0])
    config_dir = "./data/config_next/"
    config_files = os.listdir(config_dir)
    config_files = [file for file in config_files if file.endswith(".txt")]
    args = []
    for file in config_files:
        # if file.endswith("100_1000_0.1.txt"):
        # if file.split("_")[0] in ['common', 'sample', 'size_sample', 'real']:
        args.append(os.path.join(config_dir, file))
        # if file.split("_")[0] in ['gru', 'cvae', 'encore']:
        # args.append(os.path.join(config_dir, file))

    os.system('./ns3')
    with Pool(16) as pool:
        pool.map(run_baseline_parallel, args)
