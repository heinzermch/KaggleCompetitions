# Main XGBoost class, loads and processes data using external classes when necessary
# for feature engineering.
# Can do Cross-Validation or prediction on the original dataset.
# Use piplines for features
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb
import numpy as np
import sys
import operator
import time
import string
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from Manager_Skill import manager_skill
from Simple_Features import simple_features
from Neighborhood_Feature import neighborhood_feature

# Different file paths
inputDirectory = "../../../input/"
outputDirectory = "../../../output/"
trainingFile = "train.json"
testFile = "test.json"
outputFile = "submission_xgb_topmanager_topbuilding_nb.csv"
# Create the transformers
sftrans = None
mstrans = None
nbtrans = None
# Unwanted features
unwantedFeatures = ['description','photos', 'display_address', 'address1', 'desc', 'created']

def readFileFromJSON(filepath):
    df = pd.read_json(open(filepath, "r"))
    print(filepath, "has shape", df.shape)
    return df


def checkForDuplicates(word):
    word = word.lower()
    if word in ["hardwood floors","hardwood"]:
        return "hardwood"
    if word in ["laundry in building","laundry in unit","laundry room","on-site laundry","laundry","dryer in unit","washer in unit","washer/dryer"]:
        return "laundry"
    if word in ["roof deck","roof-deck","roofdeck","common roof deck"]:
        return "roof"
    if word in ["outdoor space", "common outdoor space","private outdoor space","publicoutdoor","outdoor areas"]:
        return "outdoor"
    if word in ["garden/patio", "garden", "residents garden"]:
        return "garden"
    if word in ["parking space", "common parking/garage", "parking", "garage", "on-site garage"]:
        return "parking"
    if word in ["high ceiling", "high ceilings"]:
        return "high ceiling"
    if word in ["newly renovated", "renovated"]:
        return "renovated"
    if word in ["gym/fitness", "fitness center", "gym", "fitness"]:
        return "fitness"
    if word in ["doorman", "full-time doorman"]:
        return "doorman"
    if word in ["live-in super", "live in super"]:
        return "live-in super"
    return word

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model



def splitFeaturesToMatrix(dfTrain, dfTest):

    # Treat the text in order to simplify the counting, group similar features
    for df in [dfTrain, dfTest]:
        df["features"] = df["features"].apply(lambda x: [checkForDuplicates(e) for e in x])
        df['features'] = df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))


    tfidf = CountVectorizer(stop_words='english', max_features=200)
    tr_sparse = tfidf.fit_transform(dfTrain["features"])
    te_sparse = tfidf.transform(dfTest["features"])
    return  tr_sparse, te_sparse


def toSparseMatrix(train_df, test_df):
    categorical = ["manager_id","building_id", "street_address", "neighborhood"]
    for f in categorical:
        if train_df[f].dtype == 'object':
            # print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))

    tr_sparse, te_sparse = splitFeaturesToMatrix(train_df, test_df)

    train_df.drop("features", axis=1, inplace=True)
    test_df.drop("features", axis=1, inplace=True)
    train_X = sparse.hstack([train_df, tr_sparse]).tocsr()
    test_X = sparse.hstack([test_df, te_sparse]).tocsr()
    print(train_X.shape, test_X.shape)
    return train_X, test_X


# Applying the piplines
def createFeaturesOnTrainingSet(transformers, X_train, y_train):
    for transformer in transformers:
        X_train = transformer.fit_transform(X_train, y_train)
    return X_train

def createFeaturesOnPredictionSet(transformers, X_pred):
    for transformer in transformers:
        X_pred = transformer.transform(X_pred)
    return X_pred

def initiate_transforms(doPrediction):
    sftrans = simple_features(doPrediction)
    nbtrans = neighborhood_feature()
    mstrans = manager_skill()
    return [sftrans, nbtrans, mstrans]

def run(doCV, doPrediction):
    start_time = time.time()

    transformers = initiate_transforms(doPrediction)

    print("Reading input files")
    df = readFileFromJSON(inputDirectory + trainingFile)
    # Creation of the function
    X = df.drop(["interest_level"], axis=1)
    # Changing the output to a numeric, from text
    target_num_map = {'high': 0, 'medium': 1, 'low': 2}
    y = np.array(df["interest_level"].apply(lambda x: target_num_map[x]))
    if doCV:
        # Split the dataset and create features
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
        X_train = createFeaturesOnTrainingSet(transformers, X_train, y_train)
        X_val = createFeaturesOnPredictionSet(transformers, X_val)
        print("Creating features and reading took:" + str(time.time() - start_time))
        # Drop unwanted columns:
        X_train.drop(unwantedFeatures, axis=1, inplace=True)
        X_val.drop(unwantedFeatures, axis=1, inplace=True)
        # Transform to sparse matrix format (encodes categorical variables)
        X_train, X_val = toSparseMatrix(X_train, X_val)
        # Do the training and report the logloss
        preds, model = runXGB(X_train, y_train, X_val, y_val)
        print("Full training took " + str(time.time() - start_time))
        print("log loss: " + str(log_loss(y_val, preds)))
    if doPrediction:
        X_pred = readFileFromJSON(inputDirectory + testFile)
        X = createFeaturesOnTrainingSet(transformers, X, y)
        X_pred = createFeaturesOnPredictionSet(transformers, X_pred)
        print("Creating features and reading took:" + str(time.time() - start_time))
        # Drop unwanted columns:
        X.drop(unwantedFeatures, axis=1, inplace=True)
        X_pred.drop(unwantedFeatures, axis=1, inplace=True)
        # Transform to sparse matrix format (encodes categorical variables)
        sparse_X, sparse_X_pred = toSparseMatrix(X, X_pred)
        print("Running XGB and predicting with real test set")
        preds, model = runXGB(sparse_X, y, sparse_X_pred, num_rounds=400)
        print("Full training took " + str(time.time() - start_time))
        out_df = pd.DataFrame(preds)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = X_pred.listing_id.values
        out_df.to_csv(outputDirectory + outputFile, index=False)


run(True, False)