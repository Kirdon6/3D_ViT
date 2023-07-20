import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
import point_cloud  

# Function to stack a list of ragged tensors into a single ragged tensor
def stack_ragged(tensors):
    values = tf.concat(tensors, axis=0)
    
    # Compute the lengths of each tensor in tensor
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    
    # Create a RaggedTensor
    return tf.RaggedTensor.from_row_lengths(values, lens)

# Class representing a Dataset
class Dataset:
    LABELS: int = 2

    # Inner class representing data and targets
    class Data:
        def __init__(self, data, targets) -> None:
            self.data = data
            self.targets = targets
            self.shape = self.data.shape

    def __init__(self, args: argparse.ArgumentParser, dev_size: float = 0.1, test_size: float = 0.1) -> None:
        # Load the dataset and its targets
        data, targets = point_cloud.load_dataset(args)
        
        # Split the data and targets into train, test, and dev sets
        train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=test_size, random_state=42)
        dev_size = round(dev_size / (1 - test_size), 2) 
        train_data, dev_data, train_targets, dev_targets = train_test_split(train_data, train_targets, test_size=dev_size, random_state=42)
        
        # Convert train, test, and dev sets to RaggedTensors and stack them
        train_data = stack_ragged(train_data.tolist())
        test_data = stack_ragged(test_data.tolist())
        dev_data = stack_ragged(dev_data.tolist())
        train_targets = stack_ragged(train_targets.tolist())
        test_targets = stack_ragged(test_targets.tolist())
        dev_targets = stack_ragged(dev_targets.tolist())
        
        # Create Data instances for train, test, and dev sets and set them as attributes of the Dataset object
        setattr(self, "train", self.Data(train_data, train_targets))
        setattr(self, "dev", self.Data(dev_data, dev_targets))
        setattr(self, "test", self.Data(test_data, test_targets))

    train: Data
    dev: Data
    test: Data

        