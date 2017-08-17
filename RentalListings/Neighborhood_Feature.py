# Put the listings in specific neighborhoods using pipelines.

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd
import geopandas
import geocoder
from geopandas.tools import sjoin
from shapely.geometry import Point


class neighborhood_feature(BaseEstimator, TransformerMixin):



    def __init__(self):
        self.neighborhoods = geopandas.GeoDataFrame.from_file('../../../input/nynta.shp')
        self.neighborhoods = self.neighborhoods[['NTACode','geometry']]


    def _reset(self):
        """
        No Reset necessary
        """
        return 0


    def fit(self, X, y):
        self._reset()
        # Not much to do really

        return self

    def transform(self, X):
        """
        Simple feature engineering

        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]

        """
        print("Old shape neighborhood" + str(X.shape))
        # Check for missing coordinates and replace them
        missingCoords = X[(X.longitude == 0) | (X.latitude == 0)]
        missingGeoms = (missingCoords.street_address + ' New York').apply(geocoder.google)
        X.loc[(X.longitude == 0) | (X.latitude == 0), 'latitude'] = missingGeoms.apply(lambda x: x.lat)
        X.loc[(X.longitude == 0) | (X.latitude == 0), 'longitude'] = missingGeoms.apply(lambda x: x.lng)
        # Transform to geopandas for neighborhood translation
        X['geometry'] = X.apply(lambda x: Point((float(x.longitude), float(x.latitude))), axis=1)
        apartments = geopandas.GeoDataFrame(X, geometry='geometry')
        # Set coordinate system and transofrm it to the neighborhood one
        apartments.crs = {'init': 'epsg:4326'}
        apartments.to_crs(crs=self.neighborhoods.crs, inplace=True)
        # Join or merge the two data to receive the neighborhood for each apartment
        apartments = sjoin(apartments, self.neighborhoods, how='left', op='intersects')
        # Drop unnecessary columns
        X = apartments.drop(["geometry", 'index_right'], axis=1)
        X = X.rename(columns={'NTACode': 'neighborhood'})
        print("New shape neighborhood" + str(X.shape))
        return X

