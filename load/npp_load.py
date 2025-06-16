import glob
import xarray as xr


def landscan_target_file(landscan_directory, target_date_str):
    try:
        year, month = target_date_str.split('-')[:2]
        pattern = f"{landscan_directory}/SVDNB_npp_{year}{month}*.tif"
        print(pattern)
        matches = glob.glob(pattern)
        return matches[0] if matches else None
    except Exception as e:
        print("An error occurred:", e)
        return None


def npp_read_file(landscan_dir, overpass_date):
    landscan_file = landscan_target_file(landscan_dir, overpass_date)
    landscan_ds = xr.open_dataset(landscan_file).rename({'band_data': 'npp'})
    landscan_ds.close()
    return landscan_ds


if __name__ == '__main__':
    landscan_dir = 'E:\\Date\\NPP-VIIRS'
    print(npp_read_file(landscan_dir, '2016-03-01'))