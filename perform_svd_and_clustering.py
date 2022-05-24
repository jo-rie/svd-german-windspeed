from os import makedirs
from os.path import join

import numpy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from datetime import datetime

from utils import read_nc4_file, create_geodf_from_df, svd_k_truncation, add_hour_month_season_to_df

file = 'WS_080m.2D.20099-20198.aggregated.nc4'

exportfolder = 'results/'
makedirs(exportfolder, exist_ok=True)
dims = [10, 20, 40, 60, 80, 100]
clusters = range(4, 11)
n_components = max(dims)

# approx_vals_to_compute = np.linspace(1, 101, dtype=int)
approx_vals_to_compute = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100]

seasons = ['all'] + ['winter'] + ['spring'] + ['summer'] + ['autumn']
# seasons = ['all']

for (i_season, season) in enumerate(seasons):
    print(f'{datetime.now():%H:%M:%S} - {i_season} / {len(seasons)} - Starting season {season}')
    # Load dataset
    wind_speeds, lat, lon, time = read_nc4_file(file=file, season=season)

    # Perform PCA
    u, s, v = randomized_svd(wind_speeds, n_components=n_components, random_state=None)

    print(f'{datetime.now():%H:%M:%S} - {i_season} / {len(seasons)} - Finished PCA')
    # Compute Loadings
    loadings_places = np.diag(s) @ v
    loadings_times = u @ np.diag(s)

    df_loadings_time = pd.DataFrame({
        'time': time,
        **{f'l_{i}': loadings_times[:, i] for i in range(n_components)}
    })
    df_loadings_place = pd.DataFrame({
        'lat': lat,
        'lon': lon,
        **{f'l_{i}': loadings_places[i, :] for i in range(n_components)}
    })

    print(f'{datetime.now():%H:%M:%S} - {i_season} / {len(seasons)} - Finished saving loadings')

    full_var = np.var(wind_speeds, axis=1).sum()
    expl_var = (np.diag(s) @ v).var(axis=1)
    rel_sing_vals = expl_var / full_var

    # Compute RMSE for each time
    rmse = np.full((wind_speeds.shape[0], len(approx_vals_to_compute)), fill_value=np.nan)
    for (i, approx_val) in enumerate(approx_vals_to_compute):
        approx = svd_k_truncation(u, s, v, approx_val)
        rmse[:, i] = np.sqrt(((approx - wind_speeds) ** 2).mean(axis=1))
    rmse_div_mean = rmse / np.mean(wind_speeds, axis=1, keepdims=True)

    df_approx_time = pd.DataFrame({
        'time': time,
        **{f'rmse_{l}': rmse[:, i] for (i, l) in enumerate(approx_vals_to_compute)},
        **{f'rmse_divide_mean_{l}': rmse_div_mean[:, i] for (i, l) in enumerate(approx_vals_to_compute)}
    })
    print(f'{datetime.now():%H:%M:%S} - {i_season} / {len(seasons)} - Finished RMSE time')

    # Compute RMSE for each place
    rmse = np.full((wind_speeds.shape[1], len(approx_vals_to_compute)), fill_value=np.nan)
    for (i, approx_val) in enumerate(approx_vals_to_compute):
        approx = svd_k_truncation(u, s, v, approx_val)
        rmse[:, i] = np.sqrt(((approx - wind_speeds) ** 2).mean(axis=0))
    rmse_div_mean = rmse / np.mean(wind_speeds, axis=0).reshape((-1, 1))

    df_approx_place = pd.DataFrame({
        'lat': lat,
        'lon': lon,
        **{f'rmse_{l}': rmse[:, i] for (i, l) in enumerate(approx_vals_to_compute)},
        **{f'rmse_divide_mean_{l}': rmse_div_mean[:, i] for (i, l) in enumerate(approx_vals_to_compute)}
    })
    print(f'{datetime.now():%H:%M:%S} - {i_season} / {len(seasons)} - Finished RMSE place')

    # Add hour, month and season to data frames
    add_hour_month_season_to_df(df_loadings_time)
    add_hour_month_season_to_df(df_approx_time)

    # Save everything
    print(f'{datetime.now():%H:%M:%S} - {i_season} / {len(seasons)} - Saving...')
    numpy.savez(join(exportfolder, f'numpy_data_{season}.npz'), u=u, s=s, v=v, rel_sing_vals=rel_sing_vals)
    df_loadings_time.to_pickle(join(exportfolder, f'df_loadings_time_{season}.pickle'))
    df_loadings_place.to_pickle(join(exportfolder, f'df_loadings_place_{season}.pickle'))
    df_approx_time.to_pickle(join(exportfolder, f'df_approx_time_{season}.pickle'))
    df_approx_place.to_pickle(join(exportfolder, f'df_approx_place_{season}.pickle'))

    # Perform Clusterings
    print(f'{datetime.now():%H:%M:%S} - {i_season} / {len(seasons)} - Perform Clustering')
    df_labels = pd.DataFrame({
        'lat': lat,
        'lon': lon
    })
    gdf_labels = create_geodf_from_df(df_labels)
    for nb_dimensions in dims:
        for nb_clusters in clusters:
            kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(loadings_places[:nb_dimensions, :].T)
            df_labels[f'labels_dim_{nb_dimensions}_clusters_{nb_clusters}'] = kmeans.labels_
    print(f'{datetime.now():%H:%M:%S} - {i_season} / {len(seasons)} - Saving Clustering...')
    gdf_labels.to_pickle(join(exportfolder, f'df_clusterings_{season}.pickle'))
