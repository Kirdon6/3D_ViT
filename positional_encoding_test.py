import tensorflow as tf
import math

def coordinate_encoding(coordinates, max_range, embedding_size):
    """
    Encodes 3D coordinates as positional embeddings using TensorFlow.

    Args:
        coordinates (tf.Tensor): The coordinates to encode, of shape (num_coordinates, 3).
        max_range (float): The maximum range of the coordinates.
        embedding_size (int): The size of the embedding vector.

    Returns:
        tf.Tensor: The positional embeddings of the coordinates, of shape (num_coordinates, embedding_size).
    """
    num_coordinates = tf.shape(coordinates)[0]
    position_encodings = tf.zeros([num_coordinates, embedding_size])

    # Compute the positional encodings
    for pos in range(num_coordinates):
        for i in range(0, embedding_size, 3):
            # Apply the formula for the positional encoding
            angle = pos / tf.pow(max_range, i / embedding_size)
            position_encodings[pos, i] = tf.sin(angle)
            position_encodings[pos, i + 1] = tf.cos(angle)
            position_encodings[pos, i + 2] = tf.sin(angle)

    return position_encodings

def encoding(inputs, dimension):
    inputs = tf.cast(inputs, dtype=tf.float64)
    exponent = (2 * tf.range(dimension // 2) / dimension)
    angles = 2 * math.pi * inputs / (10_000 ** exponent )
    sin_values = tf.sin(angles)
    cos_values = tf.cos(angles)
    embeddings = tf.concat([sin_values, cos_values], axis=-1)
    return embeddings



coordinates = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
dimension = 38
pos_encodings = encoding(coordinates,dimension)
print(pos_encodings)


