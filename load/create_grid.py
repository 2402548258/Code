
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd


def get_grid(xmin=73.5, ymin=18, xmax=135, ymax=53.6, width=0.1, height=0.1):
    rows = int(np.ceil((ymax-ymin) / height))
    cols = int(np.ceil((xmax-xmin) / width))
    num_of_cells = rows * cols
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax- height
    polygons = []
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom =YbottomOrigin
        for j in range(rows):
            polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)]))
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width
    grid = gpd.GeoDataFrame({'geometry':polygons}, crs = 4326)
    def getXY(pt):
        return (pt.x, pt.y)
    centroidseries = grid['geometry'].centroid
    grid['lon'],grid['lat'] = [list(t) for t in zip(*map(getXY, centroidseries))]
    List = list(range(1, (num_of_cells + 1)))
    string = 'poly'
    polygon_num = ["{}{}".format(string,i) for i in List]
    grid_area= grid['geometry'].to_crs(6933)
    grid['area'] = grid_area.area
    grid['polygon_num'] = polygon_num
    return grid
