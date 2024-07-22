###########################################################
##
## Modify paths here to point to your data sources, 
## and locations for temporary or output files
##
############################################################


# Directory for temporary or intermediate files
temp_path = '../data/2024_06_04'

# Parent data path
parent_path = '../data/2024_05_16'

# Predictors, copied from setup
predictor_features = ['NoEmp', 'CreateJob', 'LowDoc', 
       'DisbursementGross',  'new_business', 'urban_flag',
       'franchise_flag']

plot_thresh = 100


###########################################################
##
## Constants.  These do not require modification.
## These are values used across notebooks, or are
## long and placed here for convenience
##
###########################################################

#
# NN modeling parameters
#

nn_layer_sizes = [128, 64]
nn_dropout = 0.5
nn_batch_size = 32
nn_epochs = 20
nn_learning_rate = 0.0005

# Optimizer - you may want to change this based on your hardware
import tensorflow as tf
nn_optimizer = tf.keras.optimizers.legacy.Adam

# NAICS embedding sizes - vary by level
nn_naics_embed_size_dict = {'NAICS': 8,
                            'NAICS_5':8,
                            'NAICS_4':4,
                            'NAICS_3':4,
                            'NAICS_sector':2}

