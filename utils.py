from typing import Union, Dict

import cftime
import geopandas
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_nc4_file(file: str, season: str = 'all') -> Union[np.array, np.array, np.array, np.array]:
    """ Read file and extract longitudinal, latitudinal and time dependent wind speed.

    :param file: nc4 file name
    :param season: filter season from data
    :return: wind_speeds: wind speeds (len(rlat) * len(rlon)) x time in m / s
        rlon, rlat, time: np.array with longitude, latitude, and time
    """
    ds = nc.Dataset(file)
    wind_speeds = ds['wind_speed'][:]  # (time, rlat, rlon)

    rlat = ds['RLAT'][:]  # RLAT: latitude in degrees_north
    rlat = rlat.data.reshape(-1)

    rlon = ds['RLON'][:]  # RLON: longitude in degrees_east
    rlon = rlon.data.reshape(-1)

    time = cftime.num2pydate(ds['time'], units=ds['time'].units, calendar=ds['time'].calendar).data

    wind_speeds = wind_speeds.data.reshape(len(time), -1)

    if season.lower() != 'all':
        seasons = get_season(pd.Series(time))
        seasons_bool = (seasons == season.lower())
        wind_speeds = wind_speeds[seasons_bool]
        time = time[seasons_bool]

    return wind_speeds, rlat, rlon, time


def get_season(dtseries: pd.Series) -> pd.Series:
    """ Return Series of season names for Series of dates in dtseries

    :param dtseries: Series of dates
    :return: series of season names
    """
    month = dtseries.dt.month
    # map month to the meteorological season: 3-5 spring, 6-8 summer, 9-11 autumn,
    # [12, 1, 2] winter, https://www.timeanddate.com/calendar/aboutseasons.html
    season_map = 2 * ['winter'] + 3 * ['spring'] + 3 * ['summer'] + 3 * ['autumn'] + ['winter']
    return (month - 1).map(lambda s: season_map[s])


def create_geodf_from_df(df: pd.DataFrame, lon_name: str = 'lon', lat_name: str = 'lat') -> geopandas.GeoDataFrame:
    """ Create Geopandas.GeoDataFrame from df using the longitudinal and latitudinal information in columns lon_name and lat_name

    :param df: DataFrame with geographical information in columns lon_name and lat_name
    :param lon_name: column name with longitudinal information (default 'lon')
    :param lat_name: column name with latitudinal information (default 'lat')
    :return: geopandas.GeoDataFrame including columns of df
    """
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(x=df[lon_name], y=df[lat_name]))


def svd_k_truncation(u: np.array, s: np.array, v: np.array, k: np.array) -> np.array:
    """ Approximate matrix by the k-truncated singular value decomposition (u, s, v)

    :param u: matrix with left singular vectors in columns
    :param s: diagonal matrix with singular values
    :param v: matrix with right singular vectors in columns
    :param k: truncation parameter
    :return: approximation matrix of rank k
    """
    return u[:, :k] @ np.diag(s[:k]) @ v[:k, :]


def add_hour_month_season_to_df(df: pd.DataFrame, time_col_name: str = 'time') -> pd.DataFrame:
    """ Add hour, month and season column to dataframe df inferred from the column time_col_name

    :param df: DataFrame
    :param time_col_name: column name of the time column
    :return: DataFrame with additional hour, month and season column
    """
    df['hour'] = df[time_col_name].dt.hour
    df['month'] = df[time_col_name].dt.month
    df['season'] = get_season(df[time_col_name])
    return df


def scatter_points_germany(gdf: geopandas.GeoDataFrame,
                           col_to_plot: str,
                           ax: [plt.Axes, None],
                           plot_colorbar: bool = True,
                           **kwds_plot) -> plt.Axes:
    """ Create scatter plot of values in gdf[col_to_plot] over boundary of germany.

    :param gdf: geopandas data frame
    :param col_to_plot: the column of gdf to plot
    :param ax: (optional) a plt.Axes object where to plot
    :param plot_colorbar: whether to plot a colorbar besides the plot
    :param kwds_plot: passed to geopandas scatter function
    :return: plt.Axes object with plot
    """
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    germany = world[world['name'] == 'Germany']
    if ax is None:
        fig, ax = plt.subplots()
    germany.boundary.plot(edgecolor='black', ax=ax)
    ax.set(xlabel='longitude', ylabel='latitude')
    if plot_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', legend=True, cax=cax, **kwds_plot)
    else:
        gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', **kwds_plot)
    return ax
