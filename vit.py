
import tensorflow as tf

import argparse
import numpy as np
import warnings
import os
import re
import datetime
import math
from dataset import Dataset


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--create_dataset", default=False, type=bool,help="If True creates and saves dataset otherwise just loads.")
parser.add_argument("--new_file_name", default="dataset_HOLO4k", type=str, help="Path for creating dataset.")
parser.add_argument("--input_file", default="dataset_HOLO4k.npz", type=str, help="Path to saved dataset.")

parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[400], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay strength.")
parser.add_argument("--embd_size", default=75, type=float, help="Weight decay strength.")
parser.add_argument("--num_heads", default=2, type=float, help="Weight decay strength.")
parser.add_argument("--num_layers", default=2, type=float, help="Weight decay strength.")
# TODO add possible arguments

def f1_score(y_true, y_pred):
    # Convert predicted probabilities to predicted classes
    if y_pred >= 0.5:
        y_pred_classes = 1
    else:
        y_pred_classes = 0
    y_true = y_true.to_tensor()
    # Calculate true positives, false positives, and false negatives
    true_positives = tf.math.reduce_sum(y_true * y_pred_classes)
    false_positives = tf.math.reduce_sum((1 - y_true) * y_pred_classes)
    false_negatives = tf.math.reduce_sum(y_true * (1 - y_pred_classes))

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
    recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    return f1

def ragged_binary_crossentropy(y_true, y_pred):
    return tf.losses.BinaryCrossentropy()(y_true.values, y_pred)



    
class SinusoidalEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, *args, **kwargs):
        assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
        super().__init__(*args, **kwargs)
        self.dim = dim

    def get_config(self):
        return {"dim": self.dim}

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float64)
        exponent = (2 * tf.range(self.dim // 2) / self.dim)
        angles = 2 * math.pi * inputs / (10_000 ** exponent )
        sin_values = tf.sin(angles)
        cos_values = tf.cos(angles)
        embeddings = tf.concat([sin_values, cos_values], axis=-1)
        return embeddings
    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,num_heads, embd_size,  mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=75, value_dim=75) 
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_ratio * embd_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(embd_size)
        ])
        
    def call(self, inputs):
        
        residual = inputs
        
        x = self.norm1(inputs)
        x = self.self_attention(x,x)
        x += residual
        
        residual = x
        
        x = self.norm2(x)
        x = self.mlp(x)
        x += residual
        
        return x
              
class ViT3D(tf.keras.Model):
    def __init__(self, input_shape, batch_size, embd_size, num_layers, num_heads, mlp_ratio=4):
        super(ViT3D,self).__init__()
        
        self.patch_size = batch_size
        self.num_batches = (input_shape // batch_size) #* 200 #input_shape[1]
        # TODO create embeddings
        #self.pos_emb = self.add_weight("pos_emb", shape=(1, self.num_batches, embd_size))
        
        self.transformer_blocks = [TransformerBlock(num_heads,embd_size, mlp_ratio) for _ in range(num_layers)]
        
        self.normalization = tf.keras.layers.LayerNormalization()
        
        
        
    def call(self, inputs):
        
        #x = tf.reshape(inputs,(-1,self.num_batches, inputs.shape[-1]))
        x = inputs #x #+ self.pos_emb
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.normalization(x)
        x = tf.reduce_mean(x, axis=1)

        return x
    

        
        
def main(args: argparse.Namespace):
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    
        
    #TODO load dataset
    holo4k = Dataset(args, 0.33, 0.33)
    
    print(holo4k.dev.data[0].shape)

    # TODO preprocess dataset
    
    # TODO create embeddings
    
    #TODO split
    
    # TODO create input
    inputs = tf.keras.layers.Input([None,75], ragged=True)
    
    # TODO change arguments, put right data to encoder and decoder, they create positional embeddings
    transformer_output = ViT3D(holo4k.train.shape[0],args.batch_size,args.embd_size,args.num_layers,args.num_heads)(inputs)
    
    # TODO call transformer loop over layers and get output
    model = tf.keras.layers.Dense(256, activation = tf.nn.relu)(transformer_output)
    predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(model)
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    # TODO create predictions
    optimizer = tf.optimizers.experimental.AdamW(weight_decay=args.weight_decay)
    
    
    # TODO create optimizer, loss and metrics
    
    #metrics = [tf.metrics.Accuracy()]
    
    model.compile(optimizer=optimizer,
                  loss = ragged_binary_crossentropy,
                  metrics=[f1_score])
    
    model.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
    
    logs = model.fit(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=args.epochs, validation_data=(holo4k.dev.data, holo4k.dev.targets), callbacks=[model.tb_callback])
    
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}
    


           
if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in  globals() else None)
    print(main(args))
    
    
    