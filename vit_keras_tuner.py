import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import backend as K
import keras_tuner as kt
import argparse
import numpy as np
import warnings
import os
import re
import datetime
from dataset import Dataset

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--create_dataset", default=False, type=bool,help="If True creates and saves dataset otherwise just loads.")
parser.add_argument("--new_file_name", default="dataset_HOLO4k", type=str, help="Path for creating dataset.")
parser.add_argument("--proteins_path", default="holo4k", type=str, help="Path to pdb files for dataset.")
parser.add_argument("--targets_path", default="analyze_residues_holo4k", type=str, help="Path to folder with targets.")
parser.add_argument("--protein_list", default="holo4k.ds", type=str, help="Path to list of files to process")
parser.add_argument("--input_file", default="dataset_HOLO4k_small.npz", type=str, help="Path to saved dataset.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--embd_size", default=32, type=int, help="Size of atom embedding.")
parser.add_argument("--num_heads", default=8, type=int, help="Number of heads in multi-head attention.")
parser.add_argument("--num_layers", default=3, type=int, help="Number of TransformerBlocks.")
parser.add_argument("--key_dim", default=8, type=int, help="Size of key dimension for multi-head attention.")
parser.add_argument("--val_dim", default=8, type=int, help="Size of value dimension for multi-head attention.")
parser.add_argument("--learning_rate", default=1e-6, type=float, help="Initial learning rate.")

COORDINATES_SIZE = 3
CATEGORICAL_F_SIZE = 21
NEIGHBOR_F_SIZE = 8
RATIO = 15.4

# Matthews correlation coefficient
def mcc_metric(y_true, y_pred):
  predicted = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
  true_pos = tf.math.count_nonzero(predicted * y_true)
  true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
  false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
  false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
  x = tf.cast((true_pos + false_pos) * (true_pos + false_neg) 
      * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
  return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / (tf.sqrt(x) + K.epsilon())

def recall(y_true, y_pred):
    # Calculate true positives
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    # Calculate possible positives
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    # Calculate recall
    recall_keras = tp / (pos + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    # Calculate true positives
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    # Calculate predicted positives
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    # Calculate precision
    precision_keras = tp / (pred_pos + K.epsilon())
    return precision_keras


def specificity(y_true, y_pred):
    # Calculate true negatives
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    
    # Calculate false positives
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    
    # Calculate specificity
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

def ragged_weighted_binary_crossentropy(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true.values, y_pred.values, RATIO)


# Create a custom Keras layer for positional embeddings
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, *args, **kwargs):
        # Ensure the dimension is even, as it's used to create sin and cos pairs.
        assert dim % 2 == 0
        super().__init__(*args, **kwargs)
        self.dim = dim

    def get_config(self):
        return {"dim": self.dim}

    # Define the forward pass of the layer
    def call(self, inputs, batch_size, max_seq_len):
        # Calculate the boundary for separating sin and cos components.
        boundary = tf.cast(self.dim / 2, tf.int32)
        
        # Function to create the index matrix for sin values (i_smaller) and cos values (i_greater)
        def create_idx_matrix(lower_boundary, upper_boundary):
            max_sentence_len_idxs = tf.range(lower_boundary, upper_boundary, dtype=tf.float32)
            reshaped = tf.reshape(tf.repeat(max_sentence_len_idxs, repeats=max_seq_len), shape=[boundary, max_seq_len])
            transposed = tf.transpose(reshaped, perm=[1, 0])
            return transposed
        
        # Create index matrices for sin and cos components
        i_smaller = create_idx_matrix(0, boundary)
        i_greater = create_idx_matrix(boundary, self.dim)
        
        # Create a position matrix of shape [max_seq_len, boundary] for positional embeddings
        pos = tf.reshape(tf.repeat(tf.range(max_seq_len, dtype=tf.float32), boundary), shape=[max_seq_len, boundary])
        
        # Calculate the sin and cos components of positional embeddings using sine and cosine functions
        part1 = tf.math.sin(pos / (10_000 ** (2 * i_smaller / self.dim)))
        part2 = tf.math.cos(pos / (10_000 ** (2 * (i_greater - self.dim / 2) / self.dim)))
        
        # Concatenate sin and cos components along axis 1 to create the positional embeddings
        positional_embeddings = tf.concat([part1, part2], axis=1)
        
        # Tile the positional embeddings for each example in the batch
        positional_embeddings_batch = tf.reshape(tf.tile(positional_embeddings, multiples=[batch_size, 1]),
                                                 shape=[batch_size, max_seq_len, self.dim])
        
        return positional_embeddings_batch
    
    
# Define a TransformerBlock layer for the Transformer model
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, ffn_size, key_dim, value_dim, mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        
        # Layer normalization for the first sub-layer (self-attention)
        self.norm1 = tf.keras.layers.LayerNormalization()
        
        # Multi-head self-attention mechanism
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim) 
        
        # Layer normalization for the second sub-layer (feed-forward network)
        self.norm2 = tf.keras.layers.LayerNormalization()
        
        # Multi-layer perceptron (MLP) for the feed-forward network
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_ratio * ffn_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(ffn_size)
        ])
        
        # Dropout layer for regularization
        self.dropout = tf.keras.layers.Dropout(rate=args.dropout) 

    def call(self, inputs):
        # Store the input for residual connection
        residual = inputs
        
        # First sub-layer: Multi-head self-attention
        x = self.norm1(inputs)  # Apply layer normalization
        x = self.self_attention(x, x)  # Perform self-attention with the same input as query, key, and value
        x = self.dropout(x)  # Apply dropout for regularization
        x += residual  # Add the residual connection
        
        # Store the output of the first sub-layer for the second residual connection
        residual = x
        
        # Second sub-layer: Feed-forward network (MLP)
        x = self.norm2(x)  # Apply layer normalization
        x = self.mlp(x)  # Pass through the MLP
        x = self.dropout(x)  # Apply dropout for regularization
        x += residual  # Add the residual connection again
        
        return x 


# Define an Embedder layer for processing input data in a Transformer model
class Embedder(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        
        # Embedding layer for atom names
        self.atom_name_embd = tf.keras.layers.Embedding(100, dim)
        
        # Embedding layer for atom neighborhood features
        self.neighbor_embd = tf.keras.layers.Embedding(1000, dim // 8)

    # Helper function to split different parts of the input
    def split_inputs(self, inputs):
        names = inputs[:, :, :1]
        coordinates = inputs[:, :, 1:4]
        categorical = inputs[:, :, 4:25]
        numerical = inputs[:, :, 25:33]
        neighborhood = inputs[:, :, 33:41]
        return names, coordinates, categorical, numerical, neighborhood

    # Forward pass of the Embedder layer
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        names, coordinates, categorical, numerical, neighborhood = self.split_inputs(inputs)
        
        # Embed the atom names using the atom_name_embd layer
        atoms_embd = self.atom_name_embd(names)
        atoms_embd = tf.reshape(atoms_embd, shape=(batch_size, -1, atoms_embd.shape[-2] * atoms_embd.shape[-1]))
        
        categorical = tf.reshape(categorical, shape=(batch_size, -1, categorical.shape[-1]))
        
        numerical = tf.reshape(numerical, shape=(batch_size, -1, numerical.shape[-1]))

        # Embed the atom neighborhood features using the neighbor_embd layer
        neighbor_embd = self.neighbor_embd(neighborhood)
        neighbor_embd = tf.reshape(neighbor_embd, shape=(batch_size, -1, neighbor_embd.shape[-2] * neighbor_embd.shape[-1]))
        
        # Combine the atom embeddings and neighborhood embeddings using element-wise addition
        embeddings = tf.keras.layers.Add()([atoms_embd, neighbor_embd])
        
        # Concatenate all the processed features along the last dimension
        features = tf.concat([embeddings, categorical, numerical, coordinates], axis=-1)
        
        return features

        
              
# Define a 3D Vision Transformer (ViT3D) model
class ViT3D(tf.keras.Model):
    def __init__(self, num_heads, ffn_size, num_layers, key_dim, value_dim, mlp_ratio=4):
        super(ViT3D, self).__init__()
        
        # PositionalEmbedding layer to add positional information to token embeddings
        self.postional_embedding = PositionalEmbedding(ffn_size)
        
        # Create a list of TransformerBlock layers to form the transformer network
        self.transformer_blocks = [TransformerBlock(num_heads, ffn_size, key_dim, value_dim, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        
        # Layer normalization for the output of the transformer network
        self.normalization = tf.keras.layers.LayerNormalization()

    # Function to get the maximum sequence length in the input tensor
    def get_max_length(self, tensor):
        return len(tensor.row_lengths())

    # Forward pass of the ViT3D model
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        max_len = self.get_max_length(inputs)
        
        # Add positional embeddings to the input tensor
        x = inputs + self.postional_embedding(inputs, batch_size, max_len)
        
        # Apply the TransformerBlock layers sequentially to the input tensor
        for block in self.transformer_blocks:
            x = block(x)
            
        # Normalize the output of the transformer network
        x = self.normalization(x)
        
        return x  # Return the final output after passing through the transformer network
    
def model_builder(hp):
    ''' parameters to tune:
        embdedding size
        number of heads
        number of layers
        number of key dim and value dim
        mlp ratio
        learning rate
    '''
    
    
    K.clear_session()
    inputs = tf.keras.layers.Input([None,41], ragged=True)
    hp_embed_size = hp.Int('embedding', min_value = 8, max_value = 56, step = 8)
    hp_num_heads = hp.Choice('num_heads', values = [1,2,4,8,16] )
    hp_num_layers = hp.Choice('num_layers', values = [2,3,4])
    hp_kvdim = hp.Choice('kv_dim', values = [4,8,16])
    hp_mlp_ratio = hp.Choice('mlp_ratio' ,values = [1,2,4,8])    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    embeddings = Embedder(hp_embed_size)(inputs)

    transformer_output = ViT3D(hp_num_heads, hp_embed_size + CATEGORICAL_F_SIZE + NEIGHBOR_F_SIZE + COORDINATES_SIZE,hp_num_layers,hp_kvdim, hp_kvdim, hp_mlp_ratio)(embeddings)
    predictions = tf.keras.layers.Dense(1, activation=None)(transformer_output)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    optimizer = tf.optimizers.experimental.AdamW(learning_rate=hp_learning_rate)


    model.compile(optimizer=optimizer,
                  loss = ragged_weighted_binary_crossentropy,
                  metrics=[
                            tfa.metrics.F1Score(num_classes=2, average='micro', threshold=0.5),
                            mcc_metric,
                            tf.metrics.Precision(),
                            tf.metrics.Recall(),
                            tf.metrics.TrueNegatives(),
                            tf.metrics.TruePositives(),
                            tf.metrics.FalseNegatives(),
                            tf.metrics.FalsePositives(),
                            'accuracy',
                ])
    
    return model

        
def main(args: argparse.Namespace):
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    
        
    holo4k = Dataset(args)

    tuner = kt.Hyperband(model_builder, objective=kt.Objective('mcc_metric', direction='max'), max_epochs=5, factor=3, directory=args.logdir)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=args.epochs, validation_data=(holo4k.dev.data, holo4k.dev.targets), callbacks=[stop_early])
    
    #logs = model.fit(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=args.epochs, validation_data=(holo4k.dev.data, holo4k.dev.targets), callbacks=[model.tb_callback])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=args.epochs, validation_data=(holo4k.dev.data, holo4k.dev.targets))

    val_mcc_per_epoch = history.history['mcc_metric']
    best_epoch = val_mcc_per_epoch.index(max(val_mcc_per_epoch)) + 1
    
    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    logs = hypermodel.fit(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=best_epoch, validation_data=(holo4k.dev.data, holo4k.dev.targets), callbacks=[stop_early])

    hypermodel.save('best_model')
    eval_results = hypermodel.evaluate(holo4k.test.data, holo4k.test.targets)
    
    print(logs, eval_results)

    return logs, eval_results
    
    inputs = tf.keras.layers.Input([None,41], ragged=True)
    
    embeddings = Embedder(56)(inputs)
    
    transformer_output = ViT3D(16, 56 + 21 + 8 + 3,3,2, 2)(embeddings)
    predictions = tf.keras.layers.Dense(1)(transformer_output)
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    optimizer = tf.optimizers.experimental.AdamW(learning_rate=1e-4)
    
    
    model.compile(optimizer=optimizer,
                  loss = ragged_weighted_binary_crossentropy,
                  metrics=[
                            tfa.metrics.F1Score(num_classes=2, average='micro', threshold=0.5),
                            mcc_metric,
                            tf.metrics.Precision(),
                            tf.metrics.Recall(),
                            tf.metrics.TrueNegatives(),
                            tf.metrics.TruePositives(),
                            tf.metrics.FalseNegatives(),
                            tf.metrics.FalsePositives(),
                            'accuracy',
                            ])
    
    logs = model.fit(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=args.epochs, validation_data=(holo4k.dev.data, holo4k.dev.targets))
    


           
if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in  globals() else None)
    main(args)
