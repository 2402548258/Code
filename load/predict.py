import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from src.era5_processing import era_read_file
from src.cams_processing import cams_0p75_read_file_year
from src.gosat_processing import gosat_read_file
from src.npp_processing import npp_read_file
from src.odiac_processing import odiac_read_file
from src.modis_processing import modis_ndvi_read_file
from src.gfed_processing import gfed_read_file
from sklearn.preprocessing import StandardScaler
import joblib


def get_day(year):
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)

    current_date = start_date
    date_list = []
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += datetime.timedelta(days=1)

    return date_list


def predict_XCO2(cams_file, era_dir, ndvi_dir, npp_dir, trained_model, year,
                 output_dir, odiac_dir, gfed_dir, gosat_dir):
    new_lats = np.arange(18, 53.6, 0.1)
    new_lons = np.arange(73.5, 135, 0.1)
    model = joblib.load(trained_model)
    days = get_day(year)
    for d in days:
        filename = f'model_v1.0_xco2_{d}.nc'
        output_filename = os.path.join(output_dir, filename)
        if os.path.exists(output_filename):
            print(f"File already exists for {d}, skipping...")
            continue
        odiac_ds = odiac_read_file(odiac_dir, d).rename(
            {'x': 'longitude', 'y': 'latitude'}).drop_vars(['band', 'spatial_ref'])
        odiac_interp = odiac_ds.interp(latitude=new_lats, longitude=new_lons,
                                       kwargs={"fill_value": "extrapolate"})
        ndvi_ds = modis_ndvi_read_file(ndvi_dir, d)
        ndvi_interp = ndvi_ds.interp(latitude=new_lats, longitude=new_lons,
                                     kwargs={"fill_value": "extrapolate"})
        gfed_ds = gfed_read_file(gfed_dir, d)
        gfed_interp = gfed_ds.interp(latitude=new_lats, longitude=new_lons,
                                     kwargs={"fill_value": "extrapolate"})
        gosat_ds = gosat_read_file(gosat_dir, d).rename(
            {'x': 'longitude', 'y': 'latitude'}).drop_vars(['band', 'spatial_ref'])
        gosat_ds_interp = gosat_ds.interp(latitude=new_lats,
                                          longitude=new_lons,
                                          kwargs={"fill_value": "extrapolate"}
                                          )
        npp_ds = npp_read_file(npp_dir, d).rename(
            {'x': 'longitude', 'y': 'latitude'}).drop_vars(['band', 'spatial_ref'])
        npp_interp = npp_ds.interp(latitude=new_lats, longitude=new_lons,
                                   kwargs={"fill_value": "extrapolate"})
        variables_to_drop = ['expver', 'number', 'pressure_level', 'valid_time']
        era_ds = era_read_file(era_dir, d)
        existing_vars = [var for var in variables_to_drop if var in era_ds.variables]
        if existing_vars:
            era_ds = era_ds.drop_vars(existing_vars)
        era_ds = era_ds.mean(dim='valid_time')
        era_interp = era_ds.interp(latitude=new_lats, longitude=new_lons,
                                   kwargs={"fill_value": "extrapolate"})
        cams_tmp = cams_0p75_read_file_year(cams_file, d)
        cams_ds = cams_tmp.mean(dim='time')
        cams_interp = cams_ds.interp(latitude=new_lats, longitude=new_lons,
                                     kwargs={"fill_value": "extrapolate"})
        predicting_ds = xr.merge(
            [cams_interp, gosat_ds_interp, ndvi_interp, era_interp, odiac_interp, gfed_interp,
             npp_interp])
        predicting_df = predicting_ds.to_dataframe().reset_index()
        X_df = predicting_df.drop(['longitude', 'latitude', 'band', 'landscan'], axis=1)
        std = StandardScaler()
        X_tmp_df = X_df
        # 替换常见缺失标记为 NaN
        missing_markers = ["", " ", "NA", "NaN", "null", "0.0", 0, -3000]
        X_tmp_df.replace(missing_markers, np.nan, inplace=True)
        X_tmp_df = X_tmp_df.interpolate(method='cubic')
        X_tmp = std.fit_transform(X_tmp_df)
        X_tmp_df = pd.DataFrame(X_tmp, columns=X_df.columns)

        print('Predicting')
        prediction = model.predict(X_tmp_df)
        predicted_df_tmp = pd.DataFrame({
            'longitude': predicting_df['longitude'],
            'latitude': predicting_df['latitude'],
            'time': d,
            'XCO2': prediction})
        predicted_df = pd.DataFrame()
        predicted_df = pd.concat([predicted_df, predicted_df_tmp], ignore_index=True)

        predicted_ds = xr.Dataset.from_dataframe(predicted_df.set_index(['time', 'latitude', 'longitude']))
        predicted_ds['XCO2'].attrs['units'] = 'ppm'
        predicted_ds['XCO2'].attrs['long_name'] = 'Column-averaged dry-air mole fraction of CO2'
        predicted_ds['time'].attrs['long_name'] = 'Time'
        predicted_ds['longitude'].attrs['units'] = 'degrees_east'
        predicted_ds['longitude'].attrs['long_name'] = 'Longitude'
        predicted_ds['latitude'].attrs['units'] = 'degrees_north'
        predicted_ds['latitude'].attrs['long_name'] = 'Latitude'
        predicted_ds.to_netcdf(output_filename)


def main():
    era_dir = ''
    cams_file = ''
    odiac_dir = ''
    ndvi_dir = ''
    gfed_dir = ''
    npp_dir = ''
    gosat_dir = ''
    trained_model = ''
    year = []
    output = ''
    predict_XCO2(cams_file, era_dir, ndvi_dir, npp_dir, trained_model, year, output, odiac_dir,
                 gfed_dir, gosat_dir)





if __name__ == "__main__":
    main()
