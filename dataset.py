import point_cloud
import tensorflow as tf
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

def stack_ragged(tensors):
    values = tf.concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)


class Dataset:
    LABELS: int = 2
    
    class Data:
        def __init__(self,data, targets) -> None:
            self.data = data
            self.targets = targets
            self.shape = self.data.shape
        
        
    
    def __init__(self, args:argparse.ArgumentParser, dev_size: float == 0.1, test_size: float == 0.1 ) -> None:
        data, targets =  point_cloud.load_dataset(args)
        train_data, test_data, train_targets, test_targets = train_test_split(data,targets, test_size=test_size, random_state=42)
        dev_size = round(dev_size /  (1 - test_size),2)
        train_data, dev_data, train_targets, dev_targets = train_test_split(train_data, train_targets, test_size=dev_size, random_state=42)
        train_data = stack_ragged(train_data.tolist())
        test_data = stack_ragged(test_data.tolist())
        dev_data = stack_ragged(dev_data.tolist())
        train_targets = stack_ragged(train_targets.tolist())
        test_targets = stack_ragged(test_targets.tolist())
        dev_targets = stack_ragged(dev_targets.tolist())
        setattr(self, "train", self.Data(train_data, train_targets))
        setattr(self, "dev", self.Data(dev_data, dev_targets))
        setattr(self, "test", self.Data(test_data, test_targets))
        
    train:Data
    dev: Data
    test: Data
        