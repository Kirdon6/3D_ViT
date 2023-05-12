from positional_encodings.tf_encodings import TFPositionalEncoding1D
import tensorflow as tf

positional_encoding_coordinates = TFPositionalEncoding1D(3)
coords = [4.3,2.5,3.3]

print(positional_encoding_coordinates(tf.convert_to_tensor(coords)))