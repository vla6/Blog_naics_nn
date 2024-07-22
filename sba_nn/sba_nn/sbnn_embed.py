#
# Embeddigns and plots
#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

from sklearn.manifold import TSNE
from IPython.display import display, HTML

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.preprocessing import MinMaxScaler

#
# Embeddings TSNE and plot
#

def tsne_transform(embed_df):
    """ TSNE transform of embeddings data """

    trans = TSNE(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(embed_df), 
                                   index=embed_df.index)
    return emb_transformed

# TSNE plot, color by feature
def emb_color_plot(tsne_df, 
                    color_var,
                    holdout_var = 'dset_naics_holdout',
                    holdout_fill = None,
                    tsne_vars = ['tsne_0', 'tsne_1'],
                    alpha=0.7,
                    figsize = (7, 7), 
                    cmap = 'jet',
                    title_str = None,
                    title_str_2 = 'default',
                    aspect = 'equal',
                    outfile_folder = None,
                    outfile_prefix = None,
                    log_scale = False,
                    show_colorbar = True,
                    colorbar_lim = None,
                    show_axes = True,
                    xlim = None,
                    ylim = None,
                    xlabel=None,
                    ylabel=None):

    fig, ax = plt.subplots(figsize=figsize)

    s = ax.scatter(
        tsne_df[tsne_vars[0]],
        tsne_df[tsne_vars[1]],
        c=tsne_df[color_var],
        cmap=cmap,
        edgecolor = None,
        alpha=alpha,
    )
    if xlabel is None:
        ax.set_xlabel("$X_1$")
    else:
        ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel("$X_2$")
    else:
        ax.set_ylabel(ylabel)
    
    if holdout_var is not None:
        holdout_dset = tsne_df[tsne_df[holdout_var] == 1]

        if holdout_fill is None:
            hcolor = holdout_dset[color_var]
            hmap = cmap
        else:
            hcolor = holdout_fill
            hmap = None

        ax.scatter(
            holdout_dset[tsne_vars[0]],
            holdout_dset[tsne_vars[1]],
            c=hcolor,
            cmap=hmap,
            edgecolor='black')

    if title_str == '':
        plt.title(None)
    elif title_str is not None:
        if title_str_2 == 'default':
            plt.title(f'{title_str}\nby {color_var}')
        elif title_str_2 is None:
            plt.title(f'{title_str}')
        else:
            plt.title(f'{title_str}\n{title_str_2}')
    else:
        plt.title(f'TSNE by {color_var}')
    
    if colorbar_lim is not None:
        if len(colorbar_lim) == 2:
            norm = colors.Normalize(colorbar_lim[0], colorbar_lim[1])
        elif len(colorbar_lim) == 3:
            norm=colors.TwoSlopeNorm(vmin=colorbar_lim[0], vcenter=colorbar_lim[1], 
                                     vmax=colorbar_lim[2])
    elif not log_scale:
        norm = colors.Normalize(tsne_df[color_var].min(), tsne_df[color_var].max())
    else:
        norm = colors.LogNorm(tsne_df[color_var].min() + 0.002, tsne_df[color_var].max())
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect(aspect)
    
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        
    if not show_axes:
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    if outfile_folder is not None:
        fig.savefig(outfile_folder.joinpath(outfile_prefix + '_' + color_var + '.png'),
                    bbox_inches='tight')
    return ax

#
# KMeans
#

# K Means and sillhouette information
def get_clusters_silhouettes(embed_df, n_clusters, random_state = 10):
    """ Return clusers and silhouette values """
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state,
                             n_init='auto')
    cluster_labels = clusterer.fit_predict(embed_df)
    
    silhouette_avg = silhouette_score(embed_df, cluster_labels)
    sample_silhouette_values = silhouette_samples(embed_df, cluster_labels)
    
    cluster_centers = clusterer.cluster_centers_
    
    return cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values


# Plot silhouettes
# See https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
def plot_silhouette(silhouette_values, cluster_labels, label_x_pos = -0.05,
                   cmap = cm.nipy_spectral, blank_factor = 15, ax=None):
    
    n_clusters = len(np.unique(cluster_labels))
    silhouette_avg = np.mean(silhouette_values)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    xmin = np.min([-0.1, np.min(silhouette_values)])
    xmax = np.max([0.2, np.max(silhouette_values)])
    if xmax > 0.6:
        xmax = 1
    label_x_pos = np.min([label_x_pos, xmax])
    ax.set_xlim([xmin, xmax])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(silhouette_values) + (n_clusters + 1) * blank_factor])
                   
    y_lower = 10
    for index, v in np.ndenumerate(np.unique(cluster_labels)):
        i = index[0]
        
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_values[cluster_labels == v]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cmap(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(label_x_pos, y_lower + 0.5 * size_cluster_i, str(v), fontsize = 12)

        # Compute the new y_lower for next plot
        y_lower = y_upper + blank_factor  # 10 for the 0 samples

    ax.set_xlabel("Silhouette Values", fontsize = 12)
    ax.set_ylabel("Clusters", fontsize = 12)

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    

    #ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    return ax