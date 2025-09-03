import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def _set_axes_limits(ax, ks, y, yerr, num_yticks):
    kmin, kmax = int(ks.min()), int(ks.max())
    ax.set_xlim(kmin, kmax)
    # shared ticks will be set once, but we keep this for limit logic
    ax.set_xticks(np.arange(kmin, kmax + 1))

    if yerr is not None:
        vals = np.concatenate([y, y - yerr, y + yerr])
    else:
        vals = y

    finite = np.isfinite(vals)
    if finite.any():
        ymin, ymax = float(np.nanmin(vals[finite])), float(np.nanmax(vals[finite]))
    else:
        ymin, ymax = 0.0, 1.0

    if ymin == ymax:
        delta = abs(ymin) * 0.1 if ymin != 0 else 1.0
        ymin -= delta
        ymax += delta

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, num=num_yticks))


def plot_all_and_unique_metrics(
    metrics_df,
    unique_vals,
    unique_errs,
    unique_titles,
    fontsize=12,
    use_tex=True,
    linewidth=2,
    capsize=4,
    palette=None,
    num_yticks=5
):

    """
    3×3 grid with a shared X‐axis:
      Rows 0–1: 6 common metrics
      Row 2:   3 unique metrics

    Only the bottom row shows X‐ticks and X‐labels. Error bars are black;
    all lines use palette[0].
    """
    mpl.rcParams['font.size'] = fontsize
    if use_tex:
        try:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['font.family']  = 'serif'
            mpl.rcParams['font.serif']   = ['Computer Modern']
        except:
            mpl.rcParams['text.usetex'] = False

    default_palette = [
        "#0072B2", "#E69F00", "#009E73",
        "#D55E00", "#CC79A7", "#56B4E9",
        "#F0E442", "#0072B2", "#D55E00"
    ]
    if palette is None:
        palette = default_palette
    line_color = palette[0]

    # sharex=True makes all subplots use the same x‐axis limits and ticks
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    axes_flat = axes.flatten()
    ks = metrics_df['n_clusters'].to_numpy()

    common = [
            ('inertia',             'Inertia',                      False, None),
            ('silhouette',          'Silhouette Score',             False, None),
            ('calinski_harabasz',   'Calinski–Harabasz Index',      False, None),
            ('davies_bouldin',      'Davies–Bouldin Index',         False, None),
            ('avg_distance_mean',   'Avg Distance to Real Centroid',True, 'avg_distance_std'),
            ('cluster_frac_mean',   'Avg Cluster Fraction',         True, 'cluster_frac_std'),
        ]

    # Top 2 rows: common metrics
    for i, (col, title, is_err, std_col) in enumerate(common):
        ax   = axes_flat[i]
        y    = metrics_df[col].to_numpy()
        err  = metrics_df[std_col].to_numpy() if is_err else None

        if is_err:
            ax.errorbar(
                ks, y, yerr=err,
                fmt='o-', lw=linewidth, capsize=capsize,
                color=line_color, markeredgecolor=line_color,
                ecolor='black'
            )
        else:
            ax.plot(ks, y, 'o-', lw=linewidth, color=line_color)

        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True)
        _set_axes_limits(ax, ks, y, err, num_yticks)

    # Bottom row: unique metrics
    for j in range(3):
        ax   = axes_flat[6 + j]
        y    = np.array(unique_vals[j])
        err  = np.array(unique_errs[j]) if unique_errs[j] is not None else None

        ax.plot(ks, y, 'o-', lw=linewidth, color=line_color)
        if err is not None:
            ax.errorbar(
                ks, y, yerr=err, fmt='none',
                capsize=capsize, ecolor='black'
            )

        title = unique_titles[j]
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True)
        _set_axes_limits(ax, ks, y, err, num_yticks)

    # Only show X‐ticks & labels on the bottom row

    for ax in axes_flat[:6]:
        ax.tick_params(axis='x', labelbottom=False)


    for ax in axes_flat[6:]:
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('Number of clusters')



    # Explanatory text (left‐aligned, slightly larger)
    descriptions = {
        'Inertia':                       'Sum of squared distances to virtual centroids; lower → tighter clusters.',
        'Silhouette Score':              'Ranges -1 to +1; higher → well-separated clusters.',
        'Calinski–Harabasz Index':       'Higher → dense, well-separated clusters.',
        'Davies–Bouldin Index':          'Lower → better separation.',
        'Avg Distance to Real Centroid': 'Mean distance to nearest actual point centroid; lower → compact.',
        'Avg Cluster Fraction':          'Mean cluster size / total; ideal ~1/k.',
        'Cophenetic Correlation':        'Corr. between dendrogram and original distances (hierarchical).',
        'Inconsistency Mean ± Std':      'Mean±std inconsistency of merges (hierarchical).',
        'Merge-Height Mean ± Std':       'Mean±std of dendrogram merge heights (hierarchical).',
        'Gap Statistic':                 'E[log(W_ref)] – log(W_obs) (kmeans).',
        'Unbalanced Factor':             'Max/min cluster size ratio (kmeans).',
        'Avg WCSS per Cluster':          'Mean within‐cluster sum of squares (kmeans).',
        'GMM BIC':                       'Bayesian Info Criterion (GMM).',
        'GMM AIC':                       'Akaike Info Criterion (GMM).',
        'Avg Log-Lik per Component':     'Average log-likelihood per Gaussian component.'
    }

    all_titles = [t for _, t, *_ in common] + unique_titles
    lines = [
        f"{idx+1}. {title}: {descriptions.get(title, '')}"
        for idx, title in enumerate(all_titles)
    ]
    expl = "\n".join(lines)

    fig.subplots_adjust(bottom=0.28)
    fig.text(
        0.01, 0.01, expl,
        ha='left', va='bottom',
        fontsize=fontsize * 0.9,
        wrap=True,
        multialignment='left'
    )

    return fig