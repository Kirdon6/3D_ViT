import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default="dataset_HOLO4k.npz", type=str, help="Path to saved dataset.")
parser.add_argument("--new_file_name", default="dataset_HOLO4k_small", type=str, help="Path for creating dataset.")

# Script for extractin smaller complexes from dataset
if __name__ == '__main__':
    args = parser.parse_args()
    holo4k = np.load(args.input_file, allow_pickle=True)
    data = holo4k["data"]
    targets = holo4k["targets"]
    new_data = list()
    new_target = list()
    zeros = 0
    ones = 0
    for protein, target in zip(data, targets):
        if tf.shape(protein)[0] > 5000:
            continue
        else:
            unique, counts = np.unique(target, return_counts=True)
            counter = dict(zip(unique, counts))
            if 0 in counter.keys():
                zeros += counter[0]
            if 1 in counter.keys():
                ones += counter[1]
            new_data.append(protein)
            new_target.append(target)
    np.savez_compressed(args.new_file_name,data = new_data, targets=new_target)