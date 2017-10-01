#!/bin/python

import pandas as pd
from pandas import DataFrame,Series
import numpy as np
import os
import datetime

# sklearn stuff
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Imputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor

import feature_pipelines as pipes

import warnings
warnings.filterwarnings("ignore")


# ### Reading in Data
print("Reading in data...")
maindir = "/home/anerdi/Desktop/Zillow"
logerror = pd.read_csv(maindir + "/data/train_2016_v2.csv/train_2016_v2.csv")
logerror['weeknumber'] = logerror['transactiondate'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d').isocalendar()[1])
logerror['month'] = logerror['transactiondate'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d').month)
properties = pd.read_csv(maindir + "/data/properties_2016.csv/properties_2016.csv")

#proportion of living area
properties['N-LivingAreaProp'] = properties['calculatedfinishedsquarefeet']/properties['lotsizesquarefeet']

properties['N-NonLivingAreaProp'] = properties['garagetotalsqft']/properties['lotsizesquarefeet']

#Ratio of the built structure value to land area
properties['N-ValueProp'] = properties['structuretaxvaluedollarcnt']/properties['landtaxvaluedollarcnt']

#Ratio of tax of property over parcel
properties['N-ValueRatio'] = properties['taxvaluedollarcnt']/properties['taxamount']

# Pool
properties['poolsizesum'] = properties['poolsizesum'].fillna(0)
# properties['Pool'] = (properties['poolsizesum'] > 0).astype(int)
properties['Pool'] = (properties['pooltypeid2'].fillna(0) + properties['pooltypeid7'].fillna(0)).astype(int)

properties['regionidcounty'] = properties['regionidcounty'].fillna(9999)

# some more feature engineering
properties['age'] = 2017 - properties['yearbuilt']
properties['additional_rooms_count'] = np.maximum((properties['roomcnt'].values
                                                   - properties['calculatedbathnbr'].values
                                                   - properties['bedroomcnt'].values),0)


# ### Feature Pipeline
# Setup variables considered in the model

# numerical variables
num_atts = ['bedroomcnt','calculatedbathnbr','age','additional_rooms_count',
           'calculatedfinishedsquarefeet','fullbathcnt','garagecarcnt','garagetotalsqft',
            'latitude','longitude','lotsizesquarefeet', 'roomcnt',
           'numberofstories','structuretaxvaluedollarcnt','taxvaluedollarcnt','landtaxvaluedollarcnt','taxamount',
           'N-ValueRatio', 'N-LivingAreaProp', 'N-NonLivingAreaProp','N-ValueProp']

# categorical varaibles
cat_atts = ['airconditioningtypeid','architecturalstyletypeid',
           'buildingclasstypeid','heatingorsystemtypeid','Pool','propertylandusetypeid','regionidcounty',
           'storytypeid','typeconstructiontypeid','fireplaceflag','taxdelinquencyflag']

# Dictionary of categorical variables and their default levels
cat_dict = {'airconditioningtypeid':[-1] + list(range(1,14)),
           'architecturalstyletypeid':[-1] + list(range(1,28)),
           'buildingclasstypeid':[-1] + list(range(1,6)),
            'heatingorsystemtypeid':[-1] + list(range(1,26)),
            'pooltypeid10': list(range(-1,2)),
            'pooltypeid2': list(range(-1,2)),
            'pooltypeid7': list(range(-1,2)),
            'propertylandusetypeid': [-1, 31,46,47,246,247,248,260,261,262,263,264,265,266,267,268,269,270,271,
                                     273,274,275,276,279,290,291],
            'regionidcounty': [2061,3101,1286],
            'month': [-1] + list(range(1,13)),
            'storytypeid':[-1] + list(range(1,36)),
            'typeconstructiontypeid':[-1] + list(range(1,19)),
            'yearbuilt': [-1] + list(range(1885,2018)),
            'fireplaceflag': [-1] + ['True','False'],
            'taxdelinquencyflag': [-1] + ['Y','N']
           }

# Categorical pipeline
cat_pipeline = Pipeline([
        ('select_and_dummify', pipes.DF_Selector_GetDummies(cat_dict)),
    ])

# Numerical pipeline
num_pipeline = Pipeline([
        ('selector', pipes.DataFrameSelector(num_atts)),
        ('imputer', Imputer()),
    ])

# Full pipeline
feature_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])

# impute missing num_atts per regionid
for countyid in properties.regionidcounty.unique():
    # setup condition
    cond = properties['regionidcounty'] == countyid
    indices = np.where(cond)[0]
    # impute values based on region
    if countyid != 9999:
        properties.loc[indices,num_atts] = (properties.loc[indices,num_atts]
                                .fillna(properties.loc[indices,num_atts]
                                .apply(np.mean)))
    else:
        properties.loc[indices,num_atts] = (properties.loc[indices,num_atts]
                                            .fillna(properties[num_atts]
                                            .apply(np.mean)))

assert properties[num_atts].isnull().any().any() == False

# join on parcel id
data = pd.merge(properties,logerror[['parcelid','logerror','month']], on='parcelid')
data['wts_rf_10'] = np.where(data['month'] == 10, 1.5, 1)
data['wts_rf_11'] = np.where(data['month'] == 11, 1.5, 1)
data['wts_rf_12'] = np.where(data['month'] == 12, 1.5, 1)


# ### Splitting data into the K-Folds
indices = np.arange(data.shape[0])
nfolds = 100
np.random.seed(19)
np.random.shuffle(indices) # in-place shuffling

fold_indices = {(i+1):indices[i::nfolds] for i in range(nfolds)}

# ### Loading Stage 1 estimated probabilities
print("loading stage 1 estimated probabilities...")
stacked_rfs_probabilities = pd.read_csv("/home/anerdi/Desktop/Zillow/twostagemodel/overestimate_probs_stacked_rfs.csv.gz")
stacked_rfs_probabilities.rename(columns={'stacked_pred':"overestimate_prob"}, inplace=True)
stacked_rfs_probabilities = pd.merge(data[['parcelid']], stacked_rfs_probabilities, on='parcelid')

stacked_annrfs_probabilities = pd.read_csv("/home/anerdi/Desktop/Zillow/twostagemodel/overestimate_probs_stacked_ann_rfs.csv.gz")
stacked_annrfs_probabilities.rename(columns={'stacked_pred':"overestimate_prob"}, inplace=True)
stacked_annrfs_probabilities = pd.merge(data[['parcelid']], stacked_annrfs_probabilities, on='parcelid')

stacked_annrfsxgbs_probabilities = pd.read_csv("/home/anerdi/Desktop/Zillow/twostagemodel/overestimate_probs_stacked_ann_rfs_xgbs.csv.gz")
stacked_annrfsxgbs_probabilities.rename(columns={'stacked_pred':"overestimate_prob"}, inplace=True)
stacked_annrfsxgbs_probabilities = pd.merge(data[['parcelid']], stacked_annrfsxgbs_probabilities, on='parcelid')

logistic_probabiliies = pd.read_csv("/home/anerdi/Desktop/Zillow/twostagemodel/overestimate_probs.csv.gz")
logistic_probabiliies = pd.merge(data[['parcelid']], logistic_probabiliies, on='parcelid')

assert (stacked_rfs_probabilities.parcelid == data.parcelid).all()
assert (stacked_annrfs_probabilities.parcelid == data.parcelid).all()
assert (stacked_annrfsxgbs_probabilities.parcelid == data.parcelid).all()
assert (logistic_probabiliies.parcelid == data.parcelid).all()


stage1_models = [
    # ('stacked_rfs', stacked_rfs_probabilities),
    # ('stacked_annrfs', stacked_annrfs_probabilities),
    ('stacked_annrfsxgbs', stacked_annrfs_probabilities),
    # ('logistic', logistic_probabiliies)
]


# ### Training Models on the 10 splits of data \ fold_i for i = 1,...,10 & obtaining level 1 data
feature_pipeline.fit(properties) #fitting the pipeline to the entire properties dataframe

stage2_models = [
    ("rf_maxdepth8",RandomForestRegressor(n_estimators = 100, max_features= 3, random_state=9, max_depth=8, criterion='mse')),
    ("rf_maxdepth10",RandomForestRegressor(n_estimators = 100, max_features= 3, random_state=9, max_depth=10, criterion='mse')),
    ("rf_maxdepth12",RandomForestRegressor(n_estimators = 100, max_features= 3, random_state=9, max_depth=12, criterion='mse')),
]

# split training data into over/under subsets
ix_overestimated = np.where(data['logerror'] >= 0)[0]
ix_underestimated = np.where(data['logerror'] < 0)[0]
data_indices = {"over": ix_overestimated, "under": ix_underestimated}

# run loop to obtain l1 data
level_one_data = data[['parcelid']].copy()
for stage1_pair in stage1_models:
    stage1_name, stage1_probs = stage1_pair
    print("Stage 1: %s\t " % (stage1_name))

    for pair in stage2_models:
        current_model_name,current_model = pair
        print("...Current Model: %s" % (current_model_name))

        # initialize an NoneObject to be a placeholder for level-one data for current model
        model_preds = None
        print("...working on fold 1")
        for fold_nbr in range(1,nfolds+1):
            if (fold_nbr+1) % 10 == 0:
                print("...working on fold %d" % fold_nbr)

            # set training data X \ fold
            fold_trainindices = np.setdiff1d(indices,fold_indices[fold_nbr])
            fold_traindata = data.iloc[fold_trainindices,]

            # training the over/under models on their respective training data
            fold_preds_dict = {'over': None, 'under':None}
            for key,val in data_indices.items():
                type_of_zestimate, ix = key, val

                # preprocess current training data
                current_traindata = data.iloc[np.intersect1d(ix, fold_trainindices),]

                # get a clone of the model and fit the current training data
#                 print('......training model')
                reg = clone(current_model)
                reg.fit(feature_pipeline.fit_transform(current_traindata), current_traindata['logerror'])

                # level-one data (i.e., predict observations on current fold using reg)
#                 print('......obtaining level-one data')
                fold_data = data.iloc[fold_indices[fold_nbr]]
                fold_preds_overunder = Series(reg.predict(feature_pipeline.transform(fold_data)),
                                    index=fold_indices[fold_nbr], name = current_model_name)
                fold_preds_dict[type_of_zestimate] = fold_preds_overunder

            # combine over/under fold preds to get a single prediction
            fold_stage1_overestimate_probs = stage1_probs.iloc[fold_indices[fold_nbr]]['overestimate_prob']
            fold_preds = (fold_preds_dict['over']*fold_stage1_overestimate_probs
                              + fold_preds_dict['under']*(1-fold_stage1_overestimate_probs))
            fold_preds.name = stage1_name + '_' + current_model_name

            # adding to the placeholder for level-one data
            if model_preds is not None:
                model_preds = pd.concat([model_preds, fold_preds])
            else:
                model_preds = fold_preds

            # some housecleaning
            del reg

        # add level-one predictions of current model to running dataframe
        level_one_data = pd.concat([level_one_data, model_preds], axis=1)
        print("")

print("all done!")

# writing level one data to file
level_one_data.to_csv("/home/anerdi/Desktop/Zillow/levelonedata/l1data_twostage_rfs_age_stage1xgbsonly.csv.gz", index=False,
                     compression='gzip')
