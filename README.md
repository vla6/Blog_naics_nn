# Blog_naics_nn
Work for Towards Data Science 

Demonstrating random injection of "unseen" encoding values during neural network training using a custom data generator.  

## Towards Data Science [Data Disruptions to Elevate Entity Embeddings](https://towardsdatascience.com/data-disruptions-to-elevate-entity-embeddings-b1ddf86a3c95)

The version of the data for the blog post is saved in the [data_disruptions release](https://github.com/vla6/Blog_naics_nn/releases/tag/data_disruptions)

Table data is in the top level in the "tables.xlsx" document.

Code is at the top level; notebooks would run in order. Metrics are collected and summarized in 80_perf_summary.ipynb.

## Running Code

First, download the SBA Loans Dataset from Kaggle.

Then, change setup.py

  * Make input_path point to the SBA Loans dataset on your system
  * temp_path should point to a writeable directory on your system

For more information on hardware requirements and package installation, see: https://github.com/vla6/Blog_gnn_naics?tab=readme-ov-file#blog_gnn_naics
