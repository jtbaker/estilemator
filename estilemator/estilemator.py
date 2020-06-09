from joblib import Parallel
from joblib import delayed
import os
import numpy as np
import mercantile
from typing import Callable, Dict, List, Tuple, Union, Any
from affine import Affine
from rasterio.mask import geometry_mask, mask
from joblib import Parallel
from joblib import delayed
import matplotlib.pyplot as plt
import matplotlib as mpl
from rasterio import mask
import json


def get_tile_affine_transform(tile: mercantile.Tile, WIDTH=256, HEIGHT=256) -> Affine:
    bounds = mercantile.bounds(tile)
    fdw = Affine.from_gdal(
        bounds.west,
        (bounds.east - bounds.west) / WIDTH,
        0.0,
        bounds.north,
        0.0,
        -(bounds.north - bounds.south) / HEIGHT,
    )
    return fdw


# returns a 3D (WIDTH, HEIGHT, 2) shaped matrix of Longitude, Latitude pairs to make predictions on.


def get_prediction_matrix(tile: mercantile.Tile, WIDTH=256, HEIGHT=256) -> np.ndarray:
    bounds = mercantile.bounds(tile)
    x_domain = np.linspace(bounds.west, bounds.east, WIDTH)
    y_domain = np.linspace(bounds.north, bounds.south, HEIGHT)
    matrix = np.transpose(
        [np.tile(x_domain, len(y_domain)), np.repeat(y_domain, len(x_domain))]
    ).reshape((HEIGHT, WIDTH, 2))
    return matrix


class TileFactory(object):
    """
    An object you can initialize from a 2D(x,y) model such as K-Nearest 
    Neighbors to, and write the  outputs as "slippy" map tile raster images to
    a directory.
    """

    tiles: Tuple[mercantile.Tile]
    bbox: mercantile.Bbox
    estimator: Any
    zoom_levels: Union[List[int], Tuple[int]]
    transfomers: Tuple[Affine]
    tile_dimension: int

    def __init__(
        self,
        estimator,
        bbox: mercantile.Bbox,
        zoom_levels: Union[List[int], Tuple[int]],
        tile_dimension: int = 256,
    ) -> None:
        self.estimator = estimator
        self.bbox = bbox
        self.tiles = tuple(mercantile.tiles(*bbox, zooms=zoom_levels))
        self.tile_dimension = tile_dimension
        self.transformers = tuple(
            get_tile_affine_transform(tile) for tile in self.tiles
        )
        self.TILE_MATRIX_SHAPE = (
            len(self.tiles),
            self.tile_dimension,
            self.tile_dimension,
            2,
        )
        # transform to long form for predictions
        self.TILE_PREDICTION_SHAPE = (
            np.product(self.TILE_MATRIX_SHAPE[:-1]), 2)

    def generate_predictions(self, n_jobs=1, transform: callable = None, backend="multiprocessing"):
        X = np.array(
            tuple(get_prediction_matrix(tile) for tile in self.tiles)
        ).reshape(self.TILE_PREDICTION_SHAPE)
        predictions: np.ndarray = None
        if n_jobs != 1:
            with Parallel(n_jobs=n_jobs, backend=backend, prefer="threads", temp_folder="./temp") as parallel:
                delay = delayed(self.estimator.predict)
                n = len(X)
                starts = [int(n / n_jobs * i) for i in range(n_jobs)]
                ends = starts[1:] + [n]
                predictions = np.concatenate(
                    parallel(delay(X[start:end])
                             for start, end in zip(starts, ends))
                ).reshape(self.TILE_MATRIX_SHAPE[:-1])
        else:
            predictions = self.estimator.predict(
                X).reshape(self.TILE_MATRIX_SHAPE[:-1])

        if transform:
            predictions = np.array(tuple(transform(tile)
                                         for tile in predictions))
        return predictions

    def write_tiles(
        self,
        dir: str,
        n_jobs=1,
        predictions=None,
        mask_features=None,
        cmap: mpl.colors.Colormap = plt.get_cmap("viridis"),
        min_threshold: Union[int, float] = 0,
        bad_value={"color": "k", "alpha": 0},
        metadata: Dict = None,
        transform: Callable = None,
        backend="multiprocessing",
        **kwargs,
    ):
        if min_threshold is not None:
            cmap.set_bad(**bad_value)
            cmap.set_under(**bad_value)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        if metadata:
            with open("{dir}/metadata.json".format(dir=dir), "w") as file:
                file.write(json.dumps(metadata))
        if predictions is None:
            predictions = self.generate_predictions(
                n_jobs=n_jobs, transform=transform, backend=backend)
        max_threshold = np.max(predictions)
        min_threshold = np.min(predictions)
        for idx, tile in enumerate(self.tiles):
            x, y, z = tile
            preds = predictions[idx]
            if mask_features is not None:
                fdw = self.transformers[idx]
                vector_mask = geometry_mask(
                    mask_features,
                    transform=fdw,
                    all_touched=True,
                    out_shape=(self.tile_dimension, self.tile_dimension),
                )
                preds[vector_mask] = min_threshold - 100
            folder = "{directory}/{z}/{x}".format(directory=dir, z=z, x=x)
            if not os.path.exists(folder):
                os.makedirs(folder)
            file_path = "{folder}/{y}.png".format(folder=folder, y=y)

            plt.imsave(
                file_path, preds, cmap=cmap, vmin=min_threshold, vmax=max_threshold
            )
