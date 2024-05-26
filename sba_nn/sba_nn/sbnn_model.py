###########################################################
##
## Functions related to neural network modeling
## performance metrics, or other model fit information.
##
############################################################

import pandas as pd
import numpy as np
import re

import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from tensorflow.keras import layers, optimizers, losses, metrics, Model

from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, concatenate, Flatten
import tensorflow.keras.metrics as km

from keras.utils import Sequence

#
# Create model with embedding(s)
#

def create_emb_model(n_feat_base,
                 naics_max_levels,
                 naics_embedding_dim,
                 naics_embedding_names = None,
                 hidden_size = [128, 64],
                 activation='tanh', 
                 lr=0.0005,
                 opt_func = tf.keras.optimizers.legacy.Adam, 
                 dropout = 0.5):
    
    # NAICS (embedding) layers
    naics_in_layers = []
    naics_embedding_layers = []
    if naics_embedding_names is None:
        naics_embedding_names = [f'flat_naics_{x:02d}' for x in range(len(naics_max_levels))]
    
    for i in range(len(naics_max_levels)):
        input_layer = Input(shape=(1,), name=f'input_naics_{i:02d}')
        embedding_layer = Embedding(input_dim=naics_max_levels[i], 
                                    output_dim=naics_embedding_dim[i],
                                    name=f'emb_naics_{i:02d}')(input_layer)
        flatten_layer = Flatten(name = naics_embedding_names[i])(embedding_layer)
        
        naics_in_layers.append(input_layer)
        naics_embedding_layers.append(flatten_layer)
    
    # Concatenate all flattened embeddings
    concatenated_embeddings = concatenate(naics_embedding_layers)
    
    # Numeric inputs
    numeric_in_layer = Input(shape=(n_feat_base,), name='input_numeric')
        
    # Concatenate layers so far
    x = concatenate(naics_embedding_layers + [numeric_in_layer], name='input_concat')
    
    n_layers = len(hidden_size)
    
    for i in range(n_layers):
        x = Dense(hidden_size[i],activation=activation,
                  name=f'layer_{i:02d}')(x)
        x = Dropout(dropout, name=f'dropout_{i:02d}')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=[numeric_in_layer] + naics_in_layers, outputs=output)
    
    # Compile model
    optimizer = opt_func(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, 
                  metrics=[km.AUC(curve='PR'), km.AUC(curve='ROC')])
    return model

#
# Make history from fit into a legible data frame
#

def process_history(history):

    this_history_df = pd.DataFrame(history.history)
    
    # Rename columns
    try:
        this_history_df.columns = ['_'.join(c.split('_')[0:-1])  \
                                   if re.search(r'_\d+$', c) else c for c in this_history_df.columns]
    except:
        pass

    try:
        cur_col = list(this_history_df.columns)
        this_history_df.columns = [cur_col[0]] + \
            [f'{cur_col[i]}_roc'  if (cur_col[i] == cur_col[i-1]) and 'auc'in cur_col[i] \
             else cur_col[i] for i in range(1, len(cur_col))]
    except:
        pass
    
    return this_history_df

#
# Custom data generator class
# Inject missing codes into the training data at random for each epoch
#
# See also https://www.tensorflow.org/versions/r2.1/api_docs/python/tf/keras/utils/Sequence
#

    
class CatInjectGenerator(Sequence):
    def __init__(self, 
                 x_set, y_set, 
                 batch_size, 
                 categorical_columns = None, 
                 injection_rate=0.1, 
                 injection_value = 1,
                 shuffle = True):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.categorical_columns = categorical_columns
        self.injection_rate = injection_rate
        self.injection_value = injection_value
        self.indices = np.arange(len(self.x))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x.iloc[batch_indices]
        batch_y = self.y.iloc[batch_indices]

        # Separate categorical and non-categorical inputs
        batch_cat_x = [batch_x[col].values.copy() for col in self.categorical_columns]
        batch_non_cat_x = batch_x.drop(columns=self.categorical_columns).values
        
        # Inject zeros into categorical features
        if self.injection_rate > 0:
            batch_cat_x = self.inject_values(batch_cat_x)

        # Return X list and y values
        return [batch_non_cat_x] + batch_cat_x, batch_y.values

    def inject_values(self, batch_cat_x):
        for i in range(len(batch_cat_x)):
            mask = np.random.rand(*batch_cat_x[i].shape) < self.injection_rate
            batch_cat_x[i][mask] = self.injection_value
        return batch_cat_x
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)