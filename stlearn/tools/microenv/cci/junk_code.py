""" Junk code from developing the method which might come in handy later.
"""
################################################################################
                # Old version of run from analysis.py #
################################################################################

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba.typed import List

from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering
from .base import calc_neighbours, get_lrs_scores, calc_distance
from .base_grouping import get_hotspots
from .het import count, count_interactions, get_interactions

def run(adata: AnnData, lrs: np.array,
        use_label: str = None, use_het: str = 'cci_het',
        distance: int = 0, n_pairs: int = 1000, neg_binom: bool = False,
        adj_method: str = 'fdr_bh', pval_adj_cutoff: float = 0.05,
        lr_mid_dist: int = 150, min_spots: int = 5, min_expr: float = 0,
        verbose: bool = True, stats_method=False, quantile=0.05,
        plot_diagnostics: bool = False, show_plot=False,
        ):
    """Wrapper function for performing CCI analysis, varrying the analysis based
        on the inputted data / state of the anndata object.
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count.
    lrs:    np.array        The LR pairs to score/test for enrichment (in format 'L1_R1')
    use_label: str          The cell type results to use in counting.
    use_het:                The storage place for cell heterogeneity results in adata.obsm.
    distance: int           Distance to determine the neighbours (default is the nearest neighbour), distance=0 means within spot
    n_pairs: int            Number of random pairs to generate when performing the background distribution.
    neg_binom: bool         Whether to use neg-binomial distribution to estimate p-values, NOT appropriate with log1p data, alternative is to use background distribution itself (recommend higher number of n_pairs for this).
    adj_method: str         Parsed to statsmodels.stats.multitest.multipletests for multiple hypothesis testing correction.
    lr_mid_dist: int        The distance between the mid-points of the average expression of the two genes in an LR pair for it to be group with other pairs via AgglomerativeClustering to generate a common background distribution.
    min_spots: int          Minimum number of spots with an LR score to be considered for further testing.
    min_expr: float         Minimum gene expression of either L or R for spot to be considered to have reasonable score.
    Returns
    -------
    adata: AnnData          Relevant information stored: adata.uns['het'], adata.uns['lr_summary'], & data.uns['per_lr_results'].
    """
    distance = calc_distance(adata, distance)
    neighbours = calc_neighbours(adata, distance, verbose=verbose)
    adata.uns['spot_neighbours'] = pd.DataFrame([','.join(x.astype(str))
                                                           for x in neighbours],
                           index=adata.obs_names, columns=['neighbour_indices'])
    if verbose:
        print("Spot neighbour indices stored in adata.uns['spot_neighbours']")

    # Conduct with cell heterogeneity info if label_transfer provided #
    cell_het = type(use_label) != type(None) and use_label in adata.uns.keys()
    if cell_het:
        if verbose:
            print("Calculating cell hetereogeneity...")

        # Calculating cell heterogeneity #
        count(adata, distance=distance, use_label=use_label, use_het=use_het)

    het_vals = np.array([1] * len(adata)) \
                           if use_het not in adata.obsm else adata.obsm[use_het]

    """ 1. Filter any LRs without stored expression.
    """
    # Calculating the lr_scores across spots for the inputted lrs #
    lr_scores, lrs = get_lrs_scores(adata, lrs, neighbours, het_vals, min_expr)
    lr_bool = (lr_scores>0).sum(axis=0) > min_spots
    lrs = lrs[lr_bool]
    lr_scores = lr_scores[:, lr_bool]
    if verbose:
        print("Altogether " + str(len(lrs)) + " valid L-R pairs")
    if len(lrs) == 0:
        print("Exiting due to lack of valid LR pairs.")
        return

    if stats_method:
        """ Permutation based method.
          1. Group LRs with similar mean expression.
          2. Calc. common bg distrib. for grouped lrs.
          3. Calc. p-values for each lr relative to bg. 
        """
        perform_perm_testing(adata, lr_scores, n_pairs, lrs, lr_mid_dist,
                             verbose, neighbours, het_vals, min_expr,
                             neg_binom, adj_method, pval_adj_cutoff,
                            )
    else:
        """ Perform per lr background removal to get hot-spots by choosing dynamic cutoffs.
        Inspired by the watershed method:
            1. Generate set of candidate cutoffs based on quantiles.
            2. Perform DBScan clustering using spatial coordinates at different cutoffs.
            3. Choose cutoff as point that maximises number of clusters i.e. peaks. 
        """
        # TODO need to evaluate this eps param better.
        eps = 3*distance if type(distance)!=type(None) and distance!=0 else 100
        get_hotspots(adata, lr_scores.transpose(), lrs, eps=eps,
                     quantile=quantile, verbose=verbose,
                     plot_diagnostics=plot_diagnostics, show_plot=show_plot)

def perform_perm_testing(adata: AnnData, lr_scores: np.ndarray,
                         n_pairs: int, lrs: np.array,
                         lr_mid_dist: int, verbose: float, neighbours: List,
                         het_vals: np.array, min_expr: float,
                         neg_binom: bool, adj_method: str,
                         pval_adj_cutoff: float,
                         ):
    """ Performs the grouped permutation testing when taking the stats approach.
    """
    if n_pairs != 0:  # Perform permutation testing
        # Grouping spots with similar mean expression point #
        genes = get_valid_genes(adata, n_pairs)
        means_ordered, genes_ordered = get_ordered(adata, genes)
        ims = np.array(
                     [get_median_index(lr_.split('_')[0], lr_.split('_')[1],
                                        means_ordered.values, genes_ordered)
                        for lr_ in lrs]).reshape(-1, 1)

        if len(lrs) > 1: # Multi-LR pair mode, group LRs to generate backgrounds
            clusterer = AgglomerativeClustering(n_clusters=None,
                                                distance_threshold=lr_mid_dist,
                                                affinity='manhattan',
                                                linkage='single')
            lr_groups = clusterer.fit_predict(ims)
            lr_group_set = np.unique(lr_groups)
            if verbose:
                print(f'{len(lr_group_set)} lr groups with similar expression levels.')

        else: #Single LR pair mode, generate background for the LR.
            lr_groups = np.array([0])
            lr_group_set = lr_groups

        res_info = ['lr_scores', 'p_val', 'p_adj', '-log10(p_adj)',
                                                                'lr_sig_scores']
        n_, n_sigs = np.array([0]*len(lrs)), np.array([0]*len(lrs))
        per_lr_results = {}
        with tqdm(
                total=len(lr_group_set),
                desc="Generating background distributions for the LR pair groups..",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        ) as pbar:
            for group in lr_group_set:
                # Determining common mid-point for each group #
                group_bool = lr_groups==group
                group_im = int(np.median(ims[group_bool, 0]))

                # Calculating the background #
                rand_pairs = get_rand_pairs(adata, genes, n_pairs,
                                                           lrs=lrs, im=group_im)
                background = get_lrs_scores(adata, rand_pairs, neighbours,
                                            het_vals, min_expr,
                                                     filter_pairs=False).ravel()
                total_bg = len(background)
                background = background[background!=0] #Filtering for increase speed

                # Getting stats for each lr in group #
                group_lr_indices = np.where(group_bool)[0]
                for lr_i in group_lr_indices:
                    lr_ = lrs[lr_i]
                    lr_results = pd.DataFrame(index=adata.obs_names,
                                                               columns=res_info)
                    scores = lr_scores[:, lr_i]
                    stats = get_stats(scores, background, total_bg, neg_binom,
                                    adj_method, pval_adj_cutoff=pval_adj_cutoff)
                    full_stats = [scores]+list(stats)
                    for vals, colname in zip(full_stats, res_info):
                        lr_results[colname] = vals

                    n_[lr_i] = len(np.where(scores>0)[0])
                    n_sigs[lr_i] = len(np.where(
                                 lr_results['p_adj'].values<pval_adj_cutoff)[0])
                    if n_sigs[lr_i] > 0:
                        per_lr_results[lr_] = lr_results
                pbar.update(1)

        print(f"{len(per_lr_results)} LR pairs with significant interactions.")

        lr_summary = pd.DataFrame(index=lrs, columns=['n_spots', 'n_spots_sig'])
        lr_summary['n_spots'] = n_
        lr_summary['n_spots_sig'] = n_sigs
        lr_summary = lr_summary.iloc[np.argsort(-n_sigs)]

    else: #Simply store the scores
        per_lr_results = {}
        lr_summary = pd.DataFrame(index=lrs, columns=['n_spots'])
        for i, lr_ in enumerate(lrs):
            lr_results = pd.DataFrame(index=adata.obs_names,
                                                          columns=['lr_scores'])
            lr_results['lr_scores'] = lr_scores[:, i]
            per_lr_results[lr_] = lr_results
            lr_summary.loc[lr_, 'n_spots'] = len(np.where(lr_scores[:, i]>0)[0])
        lr_summary = lr_summary.iloc[np.argsort(-lr_summary.values[:,0]),:]

    adata.uns['lr_summary'] = lr_summary
    adata.uns['per_lr_results'] = per_lr_results
    if verbose:
        print("Summary of significant spots for each lr pair in adata.uns['lr_summary'].")
        print("Spot enrichment statistics of LR interactions in adata.uns['per_lr_results']")
