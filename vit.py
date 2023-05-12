import tensorflow as tf
import argparse
import numpy as np
import warnings
import os
import re
import datetime

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

parser = argparse.ArgumentParser()
# TODO add possible arguments

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,num_heads, embd_size,  mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=None, value_dim=None) 
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_ratio * embd_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers(embd_size)
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
    def __init__(self, input_shape, patch_size, embd_size, num_layers, num_heads, mlp_ratio=4):
        super(ViT3D,self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (input_shape[0] // patch_size) * input_shape[1]
        # TODO create embeddings
        self.pos_emb = self.add_weight("pos_emb", shape=(1, self.num_patches, embd_size))
        
        self.transformer_blocks = [TransformerBlock(num_heads, mlp_ratio, embd_size) for _ in range(num_layers)]
        
        self.normalization = tf.keras.layers.LayerNormalization()
        
        
        
    def call(self, inputs):
        
        x = tf.reshape(x,(-1,self.num_patches, inputs.shape[-1]))
        x = x + self.pos_emb
        
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
    data = ...
    
    
    # TODO preprocess dataset
    
    # TODO create embeddings
    
    #TODO split
    train, dev = ...
    
    # TODO create input
    inputs = tf.keras.layers.Input()
    
    # TODO change arguments, put right data to encoder and decoder, they create positional embeddings
    transformer = ViT3D(...)
    
    
    # TODO call transformer loop over layers and get output
    transformer_output = transformer(train)

    
    # TODO create predictions
    
    predictions = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(transformer_output)
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    
    
    # TODO create optimizer, loss and metrics
    
    optimizer = ...
    
    loss = ...
    
    metrics = ...
    
    model.compile(optimizer=optimizer,
                  loss = loss,
                  metrics=metrics)
    
    model.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
    
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])
    
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}
    


           
if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in  globals() else None)
    main(args)
    
    
    