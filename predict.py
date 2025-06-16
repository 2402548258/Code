import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from load.cams_load import cams_read_file
from load.era5_load import era_read_file
from load.gfed_load import gfed_read_file
from load.gosat_load import gosat_read_file
from load.modis_load import modis_ndvi_read_file
from load.npp_load import npp_read_file
from load.odiac_load import odiac_read_file
from sklearn.preprocessing import StandardScaler
import joblib


def get_dates_in_year(year):
    """Get a list of all dates (YYYY-MM-DD) in the given year"""
    start = datetime.date(year, 12, 1)
    end = datetime.date(year, 12, 31)
    return [(start + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range((end - start).days + 1)]


def interp_to_grid(ds, lats, lons, rename_dict=None, drop_vars=None):
    """Rename, drop specific variables, and interpolate dataset to a new grid"""
    if rename_dict:
        ds = ds.rename(rename_dict)
    if drop_vars:
        for v in drop_vars:
            if v in ds.variables:
                ds = ds.drop_vars(v)
    ds_interp = ds.interp(latitude=lats, longitude=lons, kwargs={"fill_value": "extrapolate"})
    return ds_interp


def preprocess_feature_dataframe(df):
    """Process feature dataframe: handle missing values and apply standard scaling"""
    missing_values = ["", " ", "NA", "NaN", "null", "0.0", 0, -3000]
    df = df.replace(missing_values, np.nan)
    df = df.interpolate(method='cubic', axis=0)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns)
    return scaled_df


def predict_XCO2(
        cams_file, era_dir, ndvi_dir, npp_dir, model_file, year,
        output_dir, odiac_dir, gfed_dir, gosat_dir):
    lat_grid = np.arange(18, 53.6, 0.1)
    lon_grid = np.arange(73.5, 135, 0.1)
    model = joblib.load(model_file)
    dates = get_dates_in_year(year)

    for date_str in dates:
        out_file = os.path.join(output_dir, f'model_v1.0_xco2_{date_str}.nc')
        if os.path.exists(out_file):
            print(f"[{date_str}] Output exists, skipping.")
            continue

        print(f"[{date_str}] Reading and interpolating source data...")
        # Read and interpolate source datasets
        odiac_interp = interp_to_grid(
            odiac_read_file(odiac_dir, date_str),
            lat_grid, lon_grid,
            rename_dict={'x': 'longitude', 'y': 'latitude'},
            drop_vars=['band', 'spatial_ref']
        )
        ndvi_interp = interp_to_grid(
            modis_ndvi_read_file(ndvi_dir, date_str),
            lat_grid, lon_grid
        )
        gfed_interp = interp_to_grid(
            gfed_read_file(gfed_dir, date_str),
            lat_grid, lon_grid
        )
        gosat_interp = interp_to_grid(
            gosat_read_file(gosat_dir, date_str),
            lat_grid, lon_grid,
            rename_dict={'x': 'longitude', 'y': 'latitude'},
            drop_vars=['band', 'spatial_ref']
        )
        npp_interp = interp_to_grid(
            npp_read_file(npp_dir, date_str),
            lat_grid, lon_grid,
            rename_dict={'x': 'longitude', 'y': 'latitude'},
            drop_vars=['band', 'spatial_ref']
        )

        # ERA5 preprocessing
        era_ds = era_read_file(era_dir, date_str)
        vars_drop = [v for v in ['expver', 'number', 'pressure_level', 'valid_time'] if v in era_ds.variables]
        if vars_drop:
            era_ds = era_ds.drop_vars(vars_drop)
        era_mean = era_ds.mean(dim='valid_time')
        era_interp = interp_to_grid(era_mean, lat_grid, lon_grid)

        # CAMS preprocessing
        cams_ds = cams_read_file(cams_file, date_str).mean(dim='time')
        cams_interp = interp_to_grid(cams_ds, lat_grid, lon_grid)

        # Merge all variables into a composite dataset
        all_interp_ds = xr.merge([
            cams_interp, gosat_interp, ndvi_interp,
            era_interp, odiac_interp, gfed_interp, npp_interp
        ])
        df = all_interp_ds.to_dataframe().reset_index()

        # Keep only major features, drop spatial and label columns
        feature_cols = [c for c in df.columns if c not in ['longitude', 'latitude', 'band', 'landscan']]
        features = preprocess_feature_dataframe(df[feature_cols])

        # Predict
        print(f"[{date_str}] Predicting XCO2 ...")
        pred_vals = model.predict(features)
        pred_df = pd.DataFrame({
            'longitude': df['longitude'],
            'latitude': df['latitude'],
            'time': date_str,
            'XCO2': pred_vals
        })

        # Convert to netCDF dataset
        print(f"[{date_str}] Writing to netCDF: {out_file}")
        pred_ds = xr.Dataset.from_dataframe(
            pred_df.set_index(['time', 'latitude', 'longitude']))
        pred_ds['XCO2'].attrs['units'] = 'ppm'
        pred_ds['XCO2'].attrs['long_name'] = 'Column-averaged dry-air mole fraction of CO2'
        pred_ds['time'].attrs['long_name'] = 'Time'
        pred_ds['longitude'].attrs.update({'units': 'degrees_east', 'long_name': 'Longitude'})
        pred_ds['latitude'].attrs.update({'units': 'degrees_north', 'long_name': 'Latitude'})

        pred_ds.to_netcdf(out_file)


def main():
    # Please specify the actual paths and parameters below
    era_dir = ''
    cams_file = ''
    odiac_dir = ''
    ndvi_dir = ''
    gfed_dir = ''
    npp_dir = ''
    gosat_dir = ''
    trained_model = ''
    year = 2016
    output_dir = ''


    predict_XCO2(
        cams_file, era_dir, ndvi_dir, npp_dir, trained_model, year, output_dir,
        odiac_dir, gfed_dir, gosat_dir
    )


if __name__ == "__main__":
    main()


