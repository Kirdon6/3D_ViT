import tensorflow as tf
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
parser.add_argument("--input_file", default="dataset_HOLO4k.npz", type=str, help="Path to saved dataset.")

parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")


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


def f1(y_true, y_pred):
    # Calculate precision and recall
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    # Calculate F1 score
    f1_score = 2 * ((prec * rec) / (prec + rec + K.epsilon()))
    return f1_score


def matthews_correlation_coefficient(y_true, y_pred):
    # Calculate true positives, true negatives, false positives, and false negatives
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    
    # Calculate Matthews Correlation Coefficient
    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = num / K.sqrt(den + K.epsilon())
    return mcc




def ragged_weighted_binary_crossentropy(y_true, y_pred):
    num_positive_examples = tf.reduce_sum(y_true.values)
    num_negative_examples = tf.reduce_sum(1 - y_true.values)
    ratio = num_negative_examples / num_positive_examples

    return tf.nn.weighted_cross_entropy_with_logits(y_true.values, y_pred.values, ratio)


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
    def __init__(self,num_heads, ffn_size, key_dim, value_dim, mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim) 
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
    def __init__(self, dim):
        super().__init__()
        self.atom_name_embd = tf.keras.layers.Embedding(100, dim)
        self.numerical_embd = tf.keras.layers.Embedding(500,dim // 8)
        
        
    def split_inputs(self, inputs):
        names = inputs[:,:,:1]
        coordinates = inputs[:,:, 1:4]
        categorical = inputs[:,:,4:25]
        numerical = inputs[:,:,25:33]
        neighborhood = inputs[:,:,33:41]
        return names, coordinates, categorical, numerical, neighborhood
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        names, coordinates, categorical,numerical, neighborhood = self.split_inputs(inputs)
        
        atoms_embd = self.atom_name_embd(names)
        atoms_embd = tf.reshape(atoms_embd, shape=( batch_size, -1, atoms_embd.shape[-2] * atoms_embd.shape[-1]))
        
        categorical = tf.reshape(categorical, shape=(batch_size,-1, categorical.shape[-1]))
        
        numerical = tf.reshape(numerical, shape=(batch_size,-1, numerical.shape[-1]))

        neighbor_embd = self.numerical_embd(neighborhood)
        neighbor_embd = tf.reshape(neighbor_embd, shape= (batch_size, -1, neighbor_embd.shape[-2] * neighbor_embd.shape[-1]))
        
        embeddings = tf.keras.layers.Add()([atoms_embd, neighbor_embd])
        features = tf.concat([embeddings, categorical, numerical, coordinates],axis=-1)

        return features
        
              
class ViT3D(tf.keras.Model):
    def __init__(self, num_heads, ffn_size, num_layers, key_dim, value_dim, mlp_ratio=4):
        super(ViT3D,self).__init__()
        self.postional_embedding = PositionalEmbedding(ffn_size)
        
        self.transformer_blocks = [TransformerBlock(num_heads,ffn_size, mlp_ratio, key_dim, value_dim) for _ in range(num_layers)]
        
        self.normalization = tf.keras.layers.LayerNormalization()
        
    def get_max_length(self, tensor):
        return len(tensor.values)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        max_len = self.get_max_length(inputs)
        
        x = inputs + self.postional_embedding(inputs, batch_size, max_len)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.normalization(x)
        return x
    
def model_builder(hp):
    ''' parameters to tune:
        embdedding size
        number of heads
        number of layers
        number of key dim and value dim
        mlp ratio
        learning rate
    '''
    
    
    
    inputs = tf.keras.layers.Input([None,41], ragged=True)
    hp_embed_size = hp.Int('embedding', min_value = 8, max_value = 56, step = 8)
    hp_num_heads = hp.Choice('num_heads', values = [1,2,4,8,16] )
    hp_num_layers = hp.Choice('num_layers', values = [2,3,4])
    hp_kvdim = hp.Choice('kv_dim', values = [4,8,16])
    hp_mlp_ratio = hp.Choice('mlp_ratio' ,values = [1,2,4,8])    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    embeddings = Embedder(hp_embed_size)(inputs)
    
    transformer_output = ViT3D(hp_num_heads, hp_embed_size + 21 + 8 + 3,hp_num_layers,hp_kvdim, hp_kvdim, hp_mlp_ratio)(embeddings)
    predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(transformer_output)
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    optimizer = tf.optimizers.experimental.AdamW(learning_rate=hp_learning_rate)
    
    
    model.compile(optimizer=optimizer,
                  loss = ragged_weighted_binary_crossentropy,
                  metrics=[
                            'accuracy',
                            precision,
                            recall,
                            f1,
                            specificity,
                            matthews_correlation_coefficient
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
    print(holo4k.train.shape)

    tuner = kt.Hyperband(model_builder, objective=kt.Objective('val_matthews_correlation_coefficient', direction='max'), max_epochs=10, factor=3, directory=args.logdir)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=args.epochs, validation_data=(holo4k.dev.data, holo4k.dev.targets), callbacks=[stop_early])
    
    #logs = model.fit(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=args.epochs, validation_data=(holo4k.dev.data, holo4k.dev.targets), callbacks=[model.tb_callback])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=args.epochs, validation_data=(holo4k.dev.data, holo4k.dev.targets))

    val_mcc_per_epoch = history.history['val_matthews_correlation_coefficient']
    best_epoch = val_mcc_per_epoch.index(max(val_mcc_per_epoch)) + 1
    
    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    logs = hypermodel.fit(holo4k.train.data, holo4k.train.targets,batch_size=args.batch_size, epochs=best_epoch, validation_data=(holo4k.dev.data, holo4k.dev.targets), callbacks=[stop_early])

    hypermodel.save('best_model')
    eval_results = hypermodel.evaluate(holo4k.test.data, holo4k.test.targets)
    
    print(logs, eval_results)

    return logs, eval_results
    


           
if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in  globals() else None)
    main(args)
    