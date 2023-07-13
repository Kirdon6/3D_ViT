import tensorflow as tf
from keras import backend as K

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
parser.add_argument("--input_file", default="dataset_HOLO4k.npz", type=str, help="Path to saved dataset.")

parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[400], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay strength.")
parser.add_argument("--num_heads", default=2, type=float, help="Weight decay strength.")
parser.add_argument("--num_layers", default=2, type=float, help="Weight decay strength.")
parser.add_argument("--embd_size", default=56, type=int, help="Weight decay strength.")
parser.add_argument("--pe_size", default=24, type=int, help="Weight decay strength.")
# TODO add possible arguments

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())



def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def ragged_binary_crossentropy(y_true, y_pred):
    return tf.losses.BinaryCrossentropy()(y_true.values, y_pred)

class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, dim, *args, **kwargs):
            assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
            super().__init__(*args, **kwargs)
            self.dim = dim

        def get_config(self):
            return {"dim": self.dim}

        def call(self, inputs, batch_size, max_seq_len):
            boundary = tf.cast(self.dim / 2 , tf.int32)
            
            def create_idx_matrix(lower_boundary, upper_boundary):
                max_sentence_len_idxs = tf.range(lower_boundary, upper_boundary, dtype=tf.float32)
                reshaped = tf.reshape(tf.repeat(max_sentence_len_idxs, repeats=max_seq_len), shape=[boundary,max_seq_len])
                transposed = tf.transpose(reshaped, perm=[1,0])
                
                return transposed
            
            i_smaller = create_idx_matrix(0,boundary)
            i_greater = create_idx_matrix(boundary, self.dim)
            
            pos = tf.reshape(tf.repeat(tf.range(max_seq_len, dtype=tf.float32), boundary), shape=[max_seq_len, boundary])
            
            part1 = tf.math.sin(pos / (10_000 ** (2 * i_smaller / self.dim)))
            part2 = tf.math.cos(pos / (10_000 ** (2 * (i_greater - self.dim/2) / self.dim)))
            positional_embeddings = tf.concat([part1, part2], axis=1)
            positional_embeddings_batch = tf.reshape(tf.tile(positional_embeddings, multiples=[batch_size, 1]),
                                                         shape=[batch_size, max_seq_len, self.dim])
            return positional_embeddings_batch
    
    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,num_heads, ffn_size,  mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=10, value_dim=10) 
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_ratio * ffn_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(ffn_size)
        ])
        self.dropout = tf.keras.layers.Dropout(rate=args.dropout)
        
    def call(self, inputs):
        
        residual = inputs
        
        x = self.norm1(inputs)
        x = self.self_attention(x,x)
        x = self.dropout(x)
        x += residual
        
        residual = x
        
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x += residual
        
        return x


class Embedder(tf.keras.layers.Layer):
    def __init__(self, dim, pe_dim):
        super().__init__()
        self.atom_name_embd = tf.keras.layers.Embedding(100, dim)
        self.numerical_embd = tf.keras.layers.Embedding(500,dim // 7)
        
        
    def split_inputs(self, inputs):
        names = inputs[:,:,:1]
        coordinates = inputs[:,:, 1:4]
        categorical = inputs[:,:,4:33]
        numerical = inputs[:,:,33:41]
        return names, coordinates, categorical, numerical
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        names, coordinates, categorical, numerical = self.split_inputs(inputs)
        
        atoms_embd = self.atom_name_embd(names)
        atoms_embd = tf.reshape(atoms_embd, shape=( batch_size, -1, atoms_embd.shape[-2] * atoms_embd.shape[-1]))
        
        categorical = tf.reshape(categorical, shape=(batch_size,-1, categorical.shape[-1]))

        num_embd = self.numerical_embd(numerical)
        num_embd = tf.reshape(num_embd, shape= (batch_size, -1, num_embd.shape[-2] * num_embd.shape[-1]))
        
        embeddings = tf.keras.layers.Add()([atoms_embd, num_embd])
        features = tf.concat([embeddings, categorical, coordinates],axis=-1)

        return features
        
              
class ViT3D(tf.keras.Model):
    def __init__(self, num_heads, ffn_size, num_layers, mlp_ratio=4):
        super(ViT3D,self).__init__()
        self.postional_embedding = PositionalEmbedding(ffn_size)
        
        self.transformer_blocks = [TransformerBlock(num_heads,ffn_size, mlp_ratio) for _ in range(num_layers)]
        
        self.normalization = tf.keras.layers.LayerNormalization()
        
    def get_max_length(self, tensor):
        #print(len(tensor.values))
        return len(tensor.values)
        
        
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        max_len = self.get_max_length(inputs)
        
        x = inputs + self.postional_embedding(inputs, batch_size, max_len)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.normalization(x)
        return x
    

        
        
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
    print(holo4k.train.shape)

    inputs = tf.keras.layers.Input([None,40], ragged=True)
    
    embeddings = Embedder(args.embd_size, args.pe_size)(inputs)
    
    transformer_output = ViT3D(args.num_heads, args.embd_size + 29 + 3,args.num_layers)(embeddings)
    predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(transformer_output)
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    optimizer = tf.optimizers.experimental.AdamW(weight_decay=args.weight_decay)
    
    
    model.compile(optimizer=optimizer,
                  loss = ragged_binary_crossentropy,
                  metrics=[
        "accuracy",
        precision,
        recall,
        f1,
        specificity,
        matthews_correlation_coefficient
    ])
    
    model.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
    
    logs = model.fit(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=args.epochs, validation_data=(holo4k.dev.data, holo4k.dev.targets), callbacks=[model.tb_callback])
    
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}
    


           
if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in  globals() else None)
    main(args)
    