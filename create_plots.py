import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from os.path import join
from os import makedirs
import sys

sys.path.append('cosmo-data')

from utils import create_geodf_from_df, read_nc4_file, scatter_points_germany

exportfolder = 'plots'
makedirs(exportfolder, exist_ok=True)
importfolder = 'results'
nc4_file = 'WS_080m.2D.20099-20198.aggregated.nc4'


# Compute export sizes for acm latex class
col_width_pts = 241.14749
col_sep_pts = 24
pts_to_inch = 1 / 72.27
col_width = col_width_pts * pts_to_inch
double_col_width = (2 * col_width_pts + col_sep_pts) * pts_to_inch
fig_height = col_width / (1.3 * 1.5)
fig_width = col_width
dpi = 600

# set Matplotlib parameters
mpl.use('pgf')
mpl.rcParams['axes.prop_cycle'] = cycler("color", plt.cm.Set2.colors)  # set color cycle
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    f"figure.figsize": (fig_width, fig_height)})
cycler_sym = mpl.colormaps['PRGn']  # plt.cm.PRGn  # Symmetric plot colors
cycler_01 = mpl.colormaps['YlGn']  # 0-1 plot colors

# Set plot parameters
n_components = 60
window_length = 90 * 24
max_sing_val = 100
seasons = ['all'] + ['winter'] + ['spring'] + ['summer'] + ['autumn']

# %% Plot singular values Over Time
rel_sing_vals = np.load(join(importfolder, 'numpy_data_all.npz'), allow_pickle=True)['rel_sing_vals']
fig, ax = plt.subplots(figsize=(fig_width, fig_height / 1.3))
ax: plt.Axes
ax.plot(np.arange(0, max_sing_val) + 1, rel_sing_vals[:max_sing_val].cumsum())
ax.set(xlabel='k')
ax.set_ylim(0, 1)
ax.axhline(y=0.9, xmax=max_sing_val, color='gray', linestyle='--')
fig.tight_layout()
fig.savefig(join(exportfolder, 'rel_sing_vals.pdf'))

# Plot singular values per season
fig, ax = plt.subplots(figsize=(fig_width, fig_height / 1.3))
ax: plt.Axes
for s in seasons[1:]:
    rel_sing_vals = np.load(join(importfolder, f'numpy_data_{s}.npz'), allow_pickle=True)['rel_sing_vals']
    ax.plot(np.arange(0, max_sing_val) + 1, rel_sing_vals[:max_sing_val].cumsum(), label=s)
ax.set(xlabel='k')
ax.set_ylim(0, 1)
ax.legend()
ax.axhline(y=0.9, xmax=max_sing_val, color='gray', linestyle='--')
fig.tight_layout()
fig.savefig(join(exportfolder, 'rel_sing_vals_seasonally.pdf'))

# %% Plot mean, variance spatially and per season
plt.switch_backend('pdf')
for season in seasons:
    wind_speeds, lat, lon, time = read_nc4_file(file=nc4_file)
    df_mean_std = pd.DataFrame({
        'lat': lat,
        'lon': lon,
        'mean': np.mean(wind_speeds, axis=0),
        'std': np.sqrt(np.var(wind_speeds, axis=0))
    })
    gdf_mean_std = create_geodf_from_df(df_mean_std)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height / 1.3))
    ax = axes[0]
    scatter_points_germany(gdf_mean_std, f'mean', ax=ax, cmap=cycler_01)
    ax.set_title('Mean [$m/s$]')
    ax = axes[1]
    scatter_points_germany(gdf_mean_std, f'std', ax=ax, cmap=cycler_01)
    ax.set_title('Standard deviation [$m/s$]')
    fig.tight_layout()
    fig.savefig(join(exportfolder, f'mean_std_spatially_season_{season}.png'), dpi=dpi)

for season in seasons:
    wind_speeds, lat, lon, time = read_nc4_file(file=nc4_file)
    df_mean_std = pd.DataFrame({
        'time': time,
        'mean': np.mean(wind_speeds, axis=1),
        'std': np.sqrt(np.var(wind_speeds, axis=1))
    })
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height * .8))
    ax.plot(df_mean_std['time'], df_mean_std['mean'].rolling(window=window_length).mean(), label='mean [$m/s$]',
            color=plt.cm.Set2.colors[0])
    ax.plot(df_mean_std['time'], df_mean_std['std'].rolling(window=window_length).mean(), label='STD [$m/s$]',
            color=plt.cm.Set2.colors[1])
    ax.legend()

    fig.tight_layout()
    fig.savefig(join(exportfolder, f'mean_std_temporally_season_{season}.pdf'))
plt.switch_backend('pgf')

# %% Plot RMSE moving average over time
df_rmse_time = pd.read_pickle(join(importfolder, 'df_approx_time_all.pickle'))
fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
ax = axes[0]
ax.plot(df_rmse_time['time'], df_rmse_time[f'rmse_{n_components}'].rolling(window=window_length).mean())
ax.set(title=r'moving average RMSE')
ax = axes[1]
ax.plot(df_rmse_time['time'], df_rmse_time[f'rmse_divide_mean_{n_components}'].rolling(window=window_length).mean())
ax.set(title=r'moving average RMSE / mean')
fig.tight_layout()
fig.savefig(join(exportfolder, f'rmse_time_mv.pdf'))

# %% Plot RMSE spatially
plt.switch_backend('pdf')
df_rmse_places = pd.read_pickle(join(importfolder, 'df_approx_place_all.pickle'))
gdf_rmse = create_geodf_from_df(df_rmse_places)
fig, ax = plt.subplots(figsize=(fig_width, fig_height / 1.3))
scatter_points_germany(gdf_rmse, f'rmse_{n_components}', ax=ax, cmap=cycler_01)
fig.tight_layout()
fig.savefig(join(exportfolder, f'rmse_places.png'), dpi=dpi)
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

# %% Plot first 4 loading plots spatially
plt.switch_backend('pdf')
df_loadings_places = pd.read_pickle(join(importfolder, 'df_loadings_place_all.pickle'))
gdf_loadings = create_geodf_from_df(df_loadings_places)
fig, axes = plt.subplots(2, 2, figsize=(col_width, col_width / 1.3))
for (i, ax) in enumerate(axes.flatten()):
    scatter_points_germany(gdf_loadings, f'l_{i + 1}', ax=ax, cmap=cycler_sym)
    ax.set(title=f'Loadings $(SV)_{i + 2}$')
fig.tight_layout()
fig.savefig(join(exportfolder, f'loadings_places.png'), dpi=dpi)
plt.switch_backend('pgf')

# %% Plot clustering
plt.switch_backend('pdf')
df_cluster = pd.read_pickle(join(importfolder, 'df_clusterings_all.pickle'))
gdf_cluster = create_geodf_from_df(df_cluster)
fig, axes = plt.subplots(2, 2, figsize=(col_width, col_width / 1.3))
for (c, ax) in zip([4, 6, 8, 10], axes.flatten()):
    scatter_points_germany(gdf_cluster, f'labels_dim_60_clusters_{c}', ax=ax, plot_colorbar=False, cmap='Set2')
    ax.set(title=f'{c} clusters')
fig.tight_layout()
fig.savefig(join(exportfolder, f'clusters.png'), dpi=dpi)
plt.switch_backend('pgf')

# %% Clustering per season
plt.switch_backend('pdf')
gdf_cluster = dict()
for s in seasons[1:]:
    df_cluster = pd.read_pickle(join(importfolder, f'df_clusterings_{s}.pickle'))
    gdf_cluster[s] = create_geodf_from_df(df_cluster)
fig, axes = plt.subplots(4, 4, figsize=(col_width, col_width / 1.3 * 1.8))
for (i, c) in enumerate([4, 6, 8, 10]):
    for (j, s) in enumerate(seasons[1:]):
        scatter_points_germany(gdf_cluster[s], f'labels_dim_60_clusters_{c}', ax=axes[i, j], plot_colorbar=False,
                               cmap='Set2')
        axes[i, j].set(title=f'{s}, {c} cl.')
fig.tight_layout()
fig.savefig(join(exportfolder, f'clusters_seasonally.png'), dpi=dpi)
plt.switch_backend('pgf')

# %% Plot hourly and seasonally boxplots
df_rmse_time = pd.read_pickle(join(importfolder, 'df_approx_time_all.pickle'))
fig, ax = plt.subplots()
df_rmse_time.boxplot(column=f'rmse_divide_mean_{n_components}', by='season', ax=ax)
ax.set_title('')
fig.tight_layout()
fig.savefig(join(exportfolder, 'seasonal_boxplot.pdf'))

fig, ax = plt.subplots()
df_rmse_time.boxplot(column=f'rmse_divide_mean_{n_components}', by='hour', ax=ax)
ax.set_title('')
fig.tight_layout()
fig.savefig(join(exportfolder, 'hourly_boxplot.pdf'))
