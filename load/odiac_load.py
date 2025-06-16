import glob
import xarray as xr


def odiac_target_file(odiac_directory, target_date_str):
    try:
        year, month = target_date_str.split('-')[:2]
        year_short = year[2:]
        pattern = f"{odiac_directory}/odiac2023_1km_excl_intl_{year_short}{month}*.tif"
        print(pattern)
        matches = glob.glob(pattern, recursive=True)
        return matches[0] if matches else None
    except Exception as e:
        print("An error occurred:", e)
        return None


def odiac_read_file(odiac_dir, overpass_date):
    odiac_file = odiac_target_file(odiac_dir, overpass_date)
    if odiac_file is None:
        raise FileNotFoundError(f"未找到日期 {overpass_date} 的 ERA 文件")
    odiac_ds = xr.open_dataset(odiac_file).rename({'band_data': 'odiac'})
    odiac_ds.close()
    return odiac_ds

if __name__ == '__main__':
    landscan_dir = 'D:\\code\\python\\jcl_xco2-main\\data\\ODIAC'
    print(odiac_read_file(landscan_dir, '2016-01-01'))
