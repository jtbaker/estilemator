
import geopandas as gpd
from sklearn.neighbors import KNeighborsRegressor, KernelDensity
import pandas as pd
import numpy as np
import pytest
import os
import shutil
import sys
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from estilemator import TileFactory

states_gdf = gpd.read_file(
    "https://raw.githubusercontent.com/jtbaker/data/master/boundaries/usstates.geojson",
    crs="EPSG:4326",
)
states_gdf = states_gdf.loc[
    ~states_gdf["NAME"].isin(["Alaska", "Hawaii", "Puerto Rico"])
]

mask_features = [
    feat.get("geometry") for feat in states_gdf.__geo_interface__.get("features")
]


def test_transform():
    CONUS_BBOX = (-131.4, 16.5, -60.6, 57.8)
    min_x, min_y, max_x, max_y = CONUS_BBOX
    x_domain = np.linspace(min_x, max_x, 1000)
    y_domain = np.linspace(min_y, max_y, 1000)
    # synthetic_data = np.random.randint(1, 100, size=10000)
    synthetic_data = np.linspace(1, 2000, 1000)
    df = pd.DataFrame({"x": x_domain, "y": y_domain, "value": synthetic_data})
    model = KNeighborsRegressor(n_neighbors=200, p=5, weights="distance")
    # model = KernelDensity()
    model.fit(df[["x", "y"]], df["value"])
    DIRECTORY = "TEST_TILE_WRITE"
    tile_factory = TileFactory(
        estimator=model, bbox=CONUS_BBOX, zoom_levels=[3, 4])
    assert len(tile_factory.transformers) == len(
        tile_factory.tiles
    ), "Tile length mismatch."
    if os.path.exists(DIRECTORY):
        shutil.rmtree(DIRECTORY)
    tile_factory.write_tiles(DIRECTORY, n_jobs=20, mask_features=mask_features)

    assert os.path.exists(DIRECTORY), "{} was not written".format(DIRECTORY)
    # shutil.rmtree(DIRECTORY)
