# Process some of the raw data and add relatively simple features

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd
import string
from scipy.stats import boxcox
import numpy as np # linear algebra


class simple_features(BaseEstimator, TransformerMixin):
    """
    Adds simple features to the reantalhop challenge data. The function should
    be usable in scikit-learn pipelines.

    Parameters
    ----------

    Attributes
    ----------
    mapping : pandas dataframe
        contains the manager_skill per manager id.

    mean_skill : float
        The mean skill of managers with at least as many listings as the
        threshold.
    """
    # Punctuation remover for text analysis
    USE_COUNTING_FEATURES = False

    def __init__(self, counting_features = False):
        self.USE_COUNTING_FEATURES = counting_features
    string.punctuation.__add__('!!')
    string.punctuation.__add__('(')
    string.punctuation.__add__(')')
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    address_map = {
        'w': 'west',
        'st.': 'street',
        'ave': 'avenue',
        'st': 'street',
        'e': 'east',
        'n': 'north',
        's': 'south'
    }

    def address_map_func(self, s):
        s = s.split(' ')
        out = []
        for x in s:
            if x in self.address_map:
                out.append(self.address_map[x])
            else:
                out.append(x)
        return ' '.join(out)


    def _reset(self):
        """
        No Reset necessary
        """
        return 0


    def fit(self, X, y):
        """Compute the skill values per manager for later use.

        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]
            The rental data. It has to contain a column named "manager_id".

        y : pandas series or numpy array, shape [n_samples]
            The corresponding target values with encoding:
            low: 0.0
            medium: 1.0
            high: 2.0
        """
        self._reset()

        # Not much to do really

        return self

    def transform(self, X):
        """
        Simple feature engineering

        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]
            Input data, has to contain "photos", "features", "price", "description", "display_address", "created", "building_id".
        """
        print("Old shape" + str(X.shape))
        X = pd.merge(left=X, right=self.oneMethodFeature(X), how='left', left_on='listing_id', right_on="listing_id", left_index=True, right_index=True)
        print("New shape" + str(X.shape))
        return X

    def createPhotoFeatures(self, df):
        # Feature engineering
        df["num_photos"] = df["photos"].apply(len)

    def createFeatureFeatures(self, df):
        df['features_count'] = df['features'].apply(lambda x: len(x))

    def createPriceFeatures(self, df):
        None
        #bc_price, tmp = boxcox(df.price)
        #df['price'] = bc_price


    def oneMethodFeature(self, oldDf):
        newDf = pd.DataFrame()
        #newDf["listing_id"] = oldDf["listing_id"]
        newDf["created"] = pd.to_datetime(oldDf["created"])
        newDf["created_year"] = newDf["created"].dt.year
        newDf["created_month"] = newDf["created"].dt.month
        newDf["created_day"] = newDf["created"].dt.day
        newDf['created_dayofweek'] = newDf["created"].dt.dayofweek
        newDf['created_dayofyear'] = newDf["created"].dt.dayofyear
        newDf['created_hour'] = newDf["created"].dt.hour
        newDf.drop("created", axis=1, inplace=True)

        newDf['zero_building_id'] = oldDf['building_id'].apply(lambda x: 1 if x == '0' else 0)

        newDf['desc'] = oldDf['description']
        newDf['desc'] = newDf['desc'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))
        newDf['desc'] = newDf['desc'].apply(lambda x: x.replace('!<br /><br />', ''))
        newDf['desc'] = newDf['desc'].apply(lambda x: x.translate(self.remove_punct_map))
        newDf['desc_letters_count'] = oldDf['description'].apply(lambda x: len(x.strip()))
        newDf['desc_words_count'] = newDf['desc'].apply(lambda x: 0 if len(x.strip()) == 0 else len(x.split(' ')))

        # Extremely Time consuming
        if self.USE_COUNTING_FEATURES:
            managers_count = oldDf['manager_id'].value_counts()
            newDf['top_10_manager'] = oldDf['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
                managers_count.values >= np.percentile(managers_count.values, 90)] else 0)
            newDf['top_25_manager'] = oldDf['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
                managers_count.values >= np.percentile(managers_count.values, 75)] else 0)
            newDf['top_5_manager'] = oldDf['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
                managers_count.values >= np.percentile(managers_count.values, 95)] else 0)
            newDf['top_50_manager'] = oldDf['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
                managers_count.values >= np.percentile(managers_count.values, 50)] else 0)
            newDf['top_1_manager'] = oldDf['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
                managers_count.values >= np.percentile(managers_count.values, 99)] else 0)
            newDf['top_2_manager'] = oldDf['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
                managers_count.values >= np.percentile(managers_count.values, 98)] else 0)
            newDf['top_15_manager'] = oldDf['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
                managers_count.values >= np.percentile(managers_count.values, 85)] else 0)
            newDf['top_20_manager'] = oldDf['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
                managers_count.values >= np.percentile(managers_count.values, 80)] else 0)
            newDf['top_30_manager'] = oldDf['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
                managers_count.values >= np.percentile(managers_count.values, 70)] else 0)

            buildings_count = oldDf['building_id'].value_counts()

            newDf['top_10_building'] = oldDf['building_id'].apply(
                lambda x: 1 if x in buildings_count.index.values[
                    buildings_count.values >= np.percentile(buildings_count.values, 90)] else 0)
            newDf['top_25_building'] = oldDf['building_id'].apply(
                lambda x: 1 if x in buildings_count.index.values[
                    buildings_count.values >= np.percentile(buildings_count.values, 75)] else 0)
            newDf['top_5_building'] = oldDf['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
                buildings_count.values >= np.percentile(buildings_count.values, 95)] else 0)
            newDf['top_50_building'] = oldDf['building_id'].apply(
                lambda x: 1 if x in buildings_count.index.values[
                    buildings_count.values >= np.percentile(buildings_count.values, 50)] else 0)
            newDf['top_1_building'] = oldDf['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
                buildings_count.values >= np.percentile(buildings_count.values, 99)] else 0)
            newDf['top_2_building'] = oldDf['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
                buildings_count.values >= np.percentile(buildings_count.values, 98)] else 0)
            newDf['top_15_building'] = oldDf['building_id'].apply(
                lambda x: 1 if x in buildings_count.index.values[
                    buildings_count.values >= np.percentile(buildings_count.values, 85)] else 0)
            newDf['top_20_building'] = oldDf['building_id'].apply(
                lambda x: 1 if x in buildings_count.index.values[
                    buildings_count.values >= np.percentile(buildings_count.values, 80)] else 0)
            newDf['top_30_building'] = oldDf['building_id'].apply(
                lambda x: 1 if x in buildings_count.index.values[
                    buildings_count.values >= np.percentile(buildings_count.values, 70)] else 0)

        newDf['address1'] = oldDf['display_address']
        newDf['address1'] = newDf['address1'].apply(lambda x: x.lower())
        newDf['address1'] = newDf['address1'].apply(lambda x: x.translate(self.remove_punct_map))
        newDf['address1'] = newDf['address1'].apply(lambda x: self.address_map_func(x))

        new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']

        for col in new_cols:
            newDf[col] = newDf['address1'].apply(lambda x: 1 if col in x else 0)
        # If the category of the address was not in the table new_cols
        newDf['other_address'] = newDf[new_cols].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)

        newDf["num_photos"] = oldDf["photos"].apply(len)
        return newDf