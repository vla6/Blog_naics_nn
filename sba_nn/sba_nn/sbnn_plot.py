##
## Plotting and metric functions for use across notebooks
##

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import textwrap


#
# Plot setups, default parameters
#


def plot_defaults():
    """ Set default plot parameters"""
    plt.style.use('seaborn-v0_8-white')
    mpl.rcParams.update({'font.size': 16})
    mpl.rcParams.update({'axes.titlesize': 18})

#
# Metric plots
#


def plot_metric_dependence(metrics, index='rate',
                          metric = 'average_precision_score',
                           title=None,
                          xformatter= None,
                          yformatter = None,
                          ylabel = None,
                          xlabel = None,
                          ax = None):
    
    use_ax = ax
    if use_ax is None:                       
        fig, ax = plt.subplots()
        
    metrics =  metrics[~metrics['dset_naics_holdout'].isna()].copy()
    metrics['dset_naics_holdout'] = metrics['dset_naics_holdout'].astype('int')
    cond = [metrics['dset_naics_holdout'] == 1, metrics['dset_naics_holdout'] == 0]
    choice=['holdout', 'test']
    metrics['dset_naics_holdout'] = np.select(cond, choice, np.nan)
                           
    met_pivot = metrics \
        .pivot(index=index, columns='dset_naics_holdout',
          values = metric) 
    
    met_pivot = met_pivot[reversed(choice)] 
    
    
    met_pivot.plot(ax=ax)
    
    ax.legend(frameon=True)
    if xformatter is not None:
        ax.xaxis.set_major_formatter(xformatter)
    if yformatter is not None:
        ax.yaxis.set_major_formatter(yformatter)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        
    ax.set_title(title)

    if use_ax == None:
        return fig



#
# History grouped plot
#

def plot_history_group(history, metric='val_auc',
                       index='epoch',
                       columns='rate',
                       xformatter= None,
                       yformatter = None,
                       ylabel = None,
                       xlabel = None,
                       col_format_func = None,
                       leg_title = None,
                      ax = None):
    
    use_ax = ax
    if use_ax is None:                       
        fig, ax = plt.subplots()
    
    piv_data = history.pivot(index=index, columns=columns, values=metric) 
    
    if col_format_func is not None:
        piv_data.columns = [col_format_func(c) for c in piv_data.columns]
    
    piv_data.plot(cmap='viridis', ax=ax)
     
    ax.legend(bbox_to_anchor=(1,1), title=leg_title, fontsize=14)
    if xformatter is not None:
        ax.xaxis.set_major_formatter(xformatter)
    if yformatter is not None:
        ax.yaxis.set_major_formatter(yformatter)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(metric)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        

    if use_ax is None:
        return fig
    
def plot_epoch_agg(history,
                   value='rate',
                   aggfunc='max',
                   xformatter= None,
                    yformatter = None,
                    ylabel = None,
                    xlabel = None,
                    ax = None):
    
    use_ax = ax
    if use_ax is None:                       
        fig, ax = plt.subplots()
    
    history.groupby('rate')['epoch'].agg('max').plot(ax=ax)

    if xformatter is not None:
        ax.xaxis.set_major_formatter(xformatter)
    if yformatter is not None:
        ax.yaxis.set_major_formatter(yformatter)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(metric)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        

    if use_ax is None:
        return fig