# Blog_naics_nn/\_A_embeddings
Work for Towards Data Science 

Visualizations and SHAP analysis of the effects random injection of "unseen" encoding values during neural network training using a custom data generator.  

![tSNE visualization of unmodified data and data with random nulls injected, showing that missing values move towards the center in embedding space with stochastic randomization](https://github.com/vla6/vla6/blob/main/images/Visualizing_Entity_Embeddings_Teaser.png)

## Visualizing Stochastic Regularization for Entity Embeddings

### Read the article at [Medium](https://medium.com/towards-data-science/visualizing-stochastic-regularization-for-entity-embeddings-c0109ced4a3a) or [TDS](https://towardsdatascience.com/visualizing-stochastic-regularization-for-entity-embeddings-c0109ced4a3a/)

Model performance data is in performance_metrics.pdf (.xlsx)

Code is at the top level; notebooks would run in order. 

##### Running Code

First, run code in the parent folder (see README.md in the parent).

Then, change A00_setup.py

  * Make temp_path point to a writeable folder in the system
  * parent_path should point to outputs from the parent folder's scripts (this is temp_path in the top-level setup) 

