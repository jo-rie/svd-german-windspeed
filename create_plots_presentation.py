import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from os.path import join
from os import makedirs
import sys
from titlecase import titlecase

sys.path.append('cosmo-data')

from utils import create_geodf_from_df, read_nc4_file, scatter_points_germany

exportfolder = '/Users/jonasrieger/Promotion/Dokumente/20220628-EDA-Vortrag/figures'
makedirs(exportfolder, exist_ok=True)
importfolder = 'results'
nc4_file = 'WS_080m.2D.20099-20198.aggregated.nc4'


def latex_math(word: str, **kwargs):
    if word.startswith('$') and word.endswith('$'):
        return word


# Compute export sizes for acm latex class
fig_width_pts = 444.14749
fig_height_pts = 232.91786
pts_to_inch = 1 / 72.27
fig_height = fig_height_pts * pts_to_inch / 1.3
fig_width = fig_width_pts * pts_to_inch
dpi = 600

# set Matplotlib parameters
mpl.use('pgf')
plt.rcParams.update({
    'axes.prop_cycle': cycler("color", plt.cm.Set2.colors),  # set color cycle
    'font.size': 8,
    'axes.labelsize': 7,
    'legend.fontsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    f"figure.figsize": (fig_width, fig_height),
    'savefig.dpi': 600})
cycler_sym = mpl.colormaps['PRGn']  # plt.cm.PRGn  # Symmetric plot colors
cycler_01 = mpl.colormaps['YlGn']  # 0-1 plot colors

# Set plot parameters
n_components = 60
window_length = 90 * 24
max_sing_val = 100
seasons = ['all'] + ['winter'] + ['spring'] + ['summer'] + ['autumn']

# %% Plot singular values Over Time
rel_sing_vals = np.load(join(importfolder, 'numpy_data_all.npz'), allow_pickle=True)['rel_sing_vals']
fig, ax = plt.subplots()
ax: plt.Axes
ax.plot(np.arange(0, max_sing_val) + 1, rel_sing_vals[:max_sing_val].cumsum())
ax.set(xlabel='$k$')
ax.set_ylim(0, 1)
ax.axhline(y=0.9, xmax=max_sing_val, color='gray', linestyle='--')
ax.set(title=titlecase('Ratio of captured inertia for given dimension $k$', callback=latex_math))
fig.tight_layout()
fig.savefig(join(exportfolder, 'rel_sing_vals.pdf'))

# Plot singular values per season
fig, ax = plt.subplots()
ax: plt.Axes
for s in seasons[1:]:
    rel_sing_vals = np.load(join(importfolder, f'numpy_data_{s}.npz'), allow_pickle=True)['rel_sing_vals']
    ax.plot(np.arange(0, max_sing_val) + 1, rel_sing_vals[:max_sing_val].cumsum(), label=s)
ax.set(xlabel='k')
ax.set_ylim(0, 1)
ax.legend()
ax.axhline(y=0.9, xmax=max_sing_val, color='gray', linestyle='--')
ax.set(title=titlecase('Ratio of captured inertia for given dimension $k$ by season'))
fig.tight_layout()
fig.savefig(join(exportfolder, 'rel_sing_vals_seasonally.pdf'))

# %% Plot mean, variance spatially and per season
plt.switch_backend('pdf')
for season in seasons:
    wind_speeds, lat, lon, time = read_nc4_file(file=nc4_file, season=season)
    df_mean_std = pd.DataFrame({
        'lat': lat,
        'lon': lon,
        'mean': np.mean(wind_speeds, axis=0),
        'std': np.sqrt(np.var(wind_speeds, axis=0))
    })
    gdf_mean_std = create_geodf_from_df(df_mean_std)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    ax = axes[0, 0]
    scatter_points_germany(gdf_mean_std, f'mean', ax=ax, cmap=cycler_01)
    ax.set_title('Spatial Mean [$m/s$]')
    ax = axes[0, 1]
    scatter_points_germany(gdf_mean_std, f'std', ax=ax, cmap=cycler_01)
    ax.set_title('Spatial Standard Deviation [$m/s$]')

    df_mean_std = pd.DataFrame({
        'time': time,
        'mean': np.mean(wind_speeds, axis=1),
        'std': np.sqrt(np.var(wind_speeds, axis=1))
    })
    ax = axes[1, 0]
    ax.plot(df_mean_std['time'], df_mean_std['mean'].rolling(window=window_length).mean(), label='mean [$m/s$]',
            color=plt.cm.Set2.colors[0])
    ax.set_title('Temporal Mean [$m / s$]')
    ax = axes[1, 1]
    ax.plot(df_mean_std['time'], df_mean_std['std'].rolling(window=window_length).mean(), label='STD [$m/s$]',
            color=plt.cm.Set2.colors[1])
    ax.set_title('Temporal Standard Deviation [$m / s$]')
    fig.tight_layout()
    fig.savefig(join(exportfolder, f'mean_std_season_{season}.png'), dpi=dpi)

plt.switch_backend('pgf')

# %% Plot Configurations per day

plt.switch_backend('pdf')
wind_speeds, lat, lon, time = read_nc4_file(file=nc4_file)

for day in range(0, 5):
    df_day = pd.DataFrame({
        'lat': lat,
        'lon': lon,
        'ws': wind_speeds[day, :]
    })
    gdf_day = create_geodf_from_df(df_day)
    fig, ax = plt.subplots()
    scatter_points_germany(gdf_day, 'ws', ax=ax, cmap=cycler_01, plot_colorbar=False)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(join(exportfolder, 'per-day', f'plot_{time[day]:%Y-%m-%d-%H}.png'), transparent=True)
    # Plot with colorbar etc
    fig, ax = plt.subplots(figsize=(fig_width * .4, fig_height))
    scatter_points_germany(gdf_day, 'ws', ax=ax, cmap=cycler_01)
    ax.set_title(f'Wind Speed at {time[day]}')
    fig.tight_layout()
    fig.savefig(join(exportfolder, 'per-day', f'plot-with-frame_{time[day]:%Y-%m-%d-%H}.png'))
plt.switch_backend('pgf')

# %% Plot local goodness of fit
plt.switch_backend('pdf')
df_rmse_time = pd.read_pickle(join(importfolder, 'df_approx_time_all.pickle'))
fig, axes = plt.subplot_mosaic([['a', 'b'], ['a', 'c']])
df_rmse_places = pd.read_pickle(join(importfolder, 'df_approx_place_all.pickle'))
gdf_rmse = create_geodf_from_df(df_rmse_places)
ax = axes['a']
scatter_points_germany(gdf_rmse, f'rmse_{n_components}', ax=ax, cmap=cycler_01)
ax.set_title('Spatial RMSE')
ax = axes['b']
ax.plot(df_rmse_time['time'], df_rmse_time[f'rmse_{n_components}'].rolling(window=window_length).mean())
ax.set(title=r'Moving Average Temporal RMSE')
ax = axes['c']
ax.plot(df_rmse_time['time'], df_rmse_time[f'rmse_divide_mean_{n_components}'].rolling(window=window_length).mean())
ax.set(title=r'Moving Average Temporal RMSE / Mean')
fig.tight_layout()
fig.savefig(join(exportfolder, f'local_gof.png'))
plt.switch_backend('pgf')

# %% Plot RMSE per season
plt.switch_backend('pdf')
fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height * 1.6))
for (i_season, season) in enumerate(seasons[1:]):
    ax = axes.flatten()[i_season]
    df_rmse_places = pd.read_pickle(join(importfolder, f'df_approx_place_{season}.pickle'))
    gdf_rmse = create_geodf_from_df(df_rmse_places)

    scatter_points_germany(gdf_rmse, f'rmse_{n_components}', ax=ax, cmap=cycler_01, vmin=.6, vmax=1.5)
    ax.set_title(f'{season.capitalize()}')
fig.tight_layout()
fig.savefig(join(exportfolder, f'rmse_places_seasonally.png'), dpi=dpi)
plt.switch_backend('pgf')

# %% Plot first 3 loading plots spatially
plt.switch_backend('pdf')
df_loadings_places = pd.read_pickle(join(importfolder, 'df_loadings_place_all.pickle'))
gdf_loadings = create_geodf_from_df(df_loadings_places)
fig, axes = plt.subplots(1, 3)
for (i, ax) in enumerate(axes.flatten()):
    scatter_points_germany(gdf_loadings, f'l_{i + 1}', ax=ax, cmap=cycler_sym)
    ax.set(title=f'Loadings $(SV)_{i + 2}$')
fig.tight_layout()
fig.savefig(join(exportfolder, f'loadings_places.png'))


fig, axes = plt.subplots(1, 2)
for (i, ax) in enumerate(axes.flatten()):
    scatter_points_germany(gdf_loadings, f'l_{i + 1}', ax=ax, cmap=cycler_sym)
    ax.set_title(f'Loadings $(SV)_{i + 2}$')
    ax.axis('off')
fig.tight_layout()
fig.savefig(join(exportfolder, f'loadings_places_transparent.png'), transparent=True)
plt.switch_backend('pgf')

# %% Plot loadings 5-13 plots spatially
plt.switch_backend('pdf')
df_loadings_places = pd.read_pickle(join(importfolder, 'df_loadings_place_all.pickle'))
gdf_loadings = create_geodf_from_df(df_loadings_places)
fig, axes = plt.subplots(2, 4)
for (i, ax) in enumerate(axes.flatten()):
    scatter_points_germany(gdf_loadings, f'l_{i + 3}', ax=ax, cmap=cycler_sym)
    ax.set(title=f'Loadings $(SV)_{i + 4}$')
fig.tight_layout()
fig.savefig(join(exportfolder, f'loadings_places_later.png'))
plt.switch_backend('pgf')

# %% Plot clustering
plt.switch_backend('pdf')
df_cluster = pd.read_pickle(join(importfolder, 'df_clusterings_all.pickle'))
gdf_cluster = create_geodf_from_df(df_cluster)
fig, axes = plt.subplots(2, 2, figsize=(fig_width/2, fig_height))
for (c, ax) in zip([4, 6, 8, 10], axes.flatten()):
    scatter_points_germany(gdf_cluster, f'labels_dim_60_clusters_{c}', ax=ax, plot_colorbar=False, cmap='Set2')
    ax.set(title=titlecase(f'{c} clusters'))
fig.tight_layout()
fig.savefig(join(exportfolder, f'clusters.png'))


# Clusters transparent
fig, ax = plt.subplots()
scatter_points_germany(gdf_cluster, f'labels_dim_60_clusters_8', ax=ax, plot_colorbar=False, cmap='Set2')
ax.axis('off')
fig.savefig(join(exportfolder, f'clusters_transparent.png'), transparent=True)
plt.switch_backend('pgf')

# %% Clustering per season
plt.switch_backend('pdf')
gdf_cluster = dict()
for s in seasons[1:]:
    df_cluster = pd.read_pickle(join(importfolder, f'df_clusterings_{s}.pickle'))
    gdf_cluster[s] = create_geodf_from_df(df_cluster)
fig, axes = plt.subplots(3, 4, figsize=(fig_width*2, fig_height*2))
for (i, c) in enumerate([6, 8, 10]):
    for (j, s) in enumerate(seasons[1:]):
        ax = axes[i, j]
        scatter_points_germany(gdf_cluster[s], f'labels_dim_60_clusters_{c}', ax=ax, plot_colorbar=False,
                               cmap='Set2')
        ax.set(title=titlecase(f'{s}, {c} cl.'))
fig.tight_layout()
fig.savefig(join(exportfolder, f'clusters_seasonally.png'))
plt.switch_backend('pgf')

# %% Plot hourly and seasonally boxplots
df_rmse_time = pd.read_pickle(join(importfolder, 'df_approx_time_all.pickle'))
fig, ax = plt.subplots()
df_rmse_time.boxplot(column=f'rmse_divide_mean_{n_components}', by='season', ax=ax)
ax.set_title('RMSE / Mean by Season')
fig.suptitle('')
fig.tight_layout()
fig.savefig(join(exportfolder, 'seasonal_boxplot.pdf'))

fig, ax = plt.subplots()
df_rmse_time.boxplot(column=f'rmse_divide_mean_{n_components}', by='hour', ax=ax)
ax.set_title('RMSE / Mean per hour of day')
fig.suptitle('')
fig.tight_layout()
fig.savefig(join(exportfolder, 'hourly_boxplot.pdf'))
