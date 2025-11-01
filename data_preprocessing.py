import os
import pandas as pd
import functools as ft
from more_itertools import sliced
from load.cams_load import cams_read_file
from load.create_grid import get_grid
from load.era5_load import era_read_file
from load.gfed_load import gfed_read_file
from load.gosat_load import gosat_read_file
from load.modis_load import modis_ndvi_read_file
from load.npp_load import npp_read_file
from load.oco_load import oco_data_files, oco_gridding
from load.odiac_load import odiac_read_file


def interp_to_df(ds, coords, drop_vars=None, rename_dims=None, fillna=False):
    """
    Interpolate a xarray Dataset on given coords and return a pandas DataFrame.
    coords: dict mapping ds dimension names to pandas Series or arrays.
    drop_vars: list of variables to drop from ds before interpolation.
    rename_dims: dict mapping dimension names in df to new column names.
    fillna: if True, fill NaNs with zeros.
    """
    ds_proc = ds
    if drop_vars:
        vars_to_drop = [v for v in drop_vars if v in ds_proc.variables]
        if vars_to_drop:
            ds_proc = ds_proc.drop_vars(vars_to_drop)
    ds_interp = ds_proc.interp(method='linear', **coords)
    df = ds_interp.to_dataframe().reset_index().drop_duplicates()
    if fillna:
        df = df.fillna(0)
    if rename_dims:
        df = df.rename(columns=rename_dims)
    return df


def prepare_training_data(
    oco_dir, era_dir, cams_file, odiac_dir,
    ndvi_dir, gfed_dir, npp_dir, gosat_dir,
    years, grid, output_file='dataset.csv', chunk_size=1000
):
    """
    Generate training dataset by merging OCO, ERA5, CAMS, ODIAC, NDVI, GFED, NPP, and GOSAT data.
    """

    # List of OCO files to process
    oco_files = oco_data_files(oco_dir, years)

    for oco_file in oco_files:
        overpass_date, oco_df = oco_gridding(oco_file, grid)
        # Read auxiliary datasets
        datasets = {
            'cams': {
                'ds': cams_read_file(cams_file, overpass_date),
                'dims': ['longitude', 'latitude', 'time'],
                'drop_vars': None,
                'rename_dims': None,
                'fillna': False
            },
            'gosat': {
                'ds': gosat_read_file(gosat_dir, overpass_date),
                'dims': ['x', 'y'],
                'drop_vars': ['spatial_ref', 'band'],
                'rename_dims': {'x': 'longitude', 'y': 'latitude'},
                'fillna': True
            },
            'ndvi': {
                'ds': modis_ndvi_read_file(ndvi_dir, overpass_date),
                'dims': ['longitude', 'latitude'],
                'drop_vars': None,
                'rename_dims': None,
                'fillna': True
            },
            'era5': {
                'ds': era_read_file(era_dir, overpass_date),
                'dims': ['longitude', 'latitude'],
                'drop_vars': ['expver', 'number', 'pressure_level', 'valid_time'],
                'rename_dims': None,
                'fillna': False
            },
            'odiac': {
                'ds': odiac_read_file(odiac_dir, overpass_date),
                'dims': ['x', 'y'],
                'drop_vars': ['spatial_ref', 'band'],
                'rename_dims': {'x': 'longitude', 'y': 'latitude'},
                'fillna': False
            },
            'gfed': {
                'ds': gfed_read_file(gfed_dir, overpass_date),
                'dims': ['longitude', 'latitude'],
                'drop_vars': None,
                'rename_dims': None,
                'fillna': True
            },
            'npp': {
                'ds': npp_read_file(npp_dir, overpass_date),
                'dims': ['x', 'y'],
                'drop_vars': ['spatial_ref', 'band'],
                'rename_dims': {'x': 'longitude', 'y': 'latitude'},
                'fillna': True
            }
        }

        # Process OCO data in chunks
        for idx_slice in sliced(range(len(oco_df)), chunk_size):
            chunk = oco_df.iloc[idx_slice].reset_index(drop=True)
            chunk['time'] = pd.to_datetime(chunk['rounded_time'])

            # Coordinate mapping for interpolation
            coords_map = {
                'longitude': chunk['longitude'],
                'latitude': chunk['latitude'],
                'time': chunk['time']
            }

            df_list = [chunk]

            # Interpolate each dataset and append to list
            for cfg in datasets.values():
                coords = {
                    dim: coords_map['longitude'] if dim in ['x', 'longitude'] else
                         coords_map['latitude'] if dim in ['y', 'latitude'] else
                         coords_map['time']
                    for dim in cfg['dims']
                }
                df = interp_to_df(
                    cfg['ds'], coords,
                    drop_vars=cfg['drop_vars'],
                    rename_dims=cfg['rename_dims'],
                    fillna=cfg['fillna']
                )
                df_list.append(df)

            # Merge all DataFrames on longitude and latitude
            merged = ft.reduce(lambda left, right: pd.merge(left, right, on=['longitude', 'latitude']), df_list)

            # Drop unwanted columns
            for col in ['index']:
                if col in merged.columns:
                    merged = merged.drop(columns=col)

            # Write to CSV
            merged.drop(merged.columns[[4,5,7,11,20,23]], axis=1, inplace=True)
            merged.to_csv(output_file, mode='a', header=False, index=False)

    print(f'Training data generated and saved to {output_file}')


def main():
    oco_dir = ''
    years = []
    grid = get_grid()
    era_dir = ''
    cams_file = ''
    odiac_dir = ''
    ndvi_dir = ''
    gfed_dir = ''
    npp_dir = ''
    gosat_dir = ''


    prepare_training_data(
        oco_dir, era_dir, cams_file,
        odiac_dir, ndvi_dir,
        gfed_dir, npp_dir, gosat_dir,
        years, grid
    )

if __name__ == '__main__':
    main()

