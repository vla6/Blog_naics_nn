###########################################################
##
## Class definition for imputer to transform 
## tabular data to a format suitable for NN.
## Includes mean fill and data scaling
##
## To enable NAICS (categorical) features which will later have
## entity embeddings, I shift OrdinalEncoder values 
## so that missing or unknown values get 0/1 encoded 
## values.  Note that if I try to use 0 or 1 as for
## unseens in raw OrdinalEncoder, I will get an error
##
############################################################

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, OrdinalEncoder

reserve_naics_encoding_levels = 2

class NNImputer():
    """ Class that contains several sklearn imputers to convert 
    tabular small business administration data to a format that
    is suitable for the GNN.  Specifically, missing values are
    median filled, (probably) continuous features are quantile scaled, 
    and remaining low-cardinality features are MinMax scaled.  """
    
    # set in fit
    max_naics_cat = None

    def scaler_trans_append(self, data_in, data_trans, scaler):
        """Post-processing of data output by a scaler, to ensure we keep all
        fields, not just those scaled.  """
        
        trans_df = pd.DataFrame(data_trans,
                                columns=scaler.feature_names_in_)
        trans_df.index = data_in.index
        
        # Get columns not affected by the transformation
        fix_cols = [c for c in data_in.columns if c not in trans_df.columns]
        
        # Return recombined data with same column order
        return pd.concat([data_in[fix_cols], trans_df], axis=1)[data_in.columns]

    def median_imputer_trans_to_pandas(self, mat, index=None):
        """ Takes in matrix-format data output from the median imputer, and 
        converts to pandas.  Optionally add an index"""
        return pd.DataFrame(mat, columns= self.median_imputer.get_feature_names_out(),
                           index = index)
    
    def median_imputer_create_fit(self, data, transform=True):
        """ Create and fit the median imputer.  Optionally also transform the
        input data as it will be used for scaling"""
        
        
        self.median_imputer = SimpleImputer(missing_values=np.nan, 
                                            strategy='median',
                                            add_indicator=True)  
        if transform:
            data = data[self.features_in + self.naics_features]
            data_trans = \
                self.median_imputer_trans_to_pandas(self.median_imputer \
                                               .fit_transform(data[self.features_in]),
                                                   index = data.index)
            fix_cols = [c for c in data.columns if c not in data_trans.columns]
            return pd.concat([data_trans, data[fix_cols]], axis=1)

        else:
            self.median_imputer.fit(data[self.features_in])
            return None
        
    def median_imputer_transform(self, data):
        """ Transform data using the median imputer, returning
        Pandas data"""
        data = data[self.features_in + self.naics_features]
        data_trans = self.median_imputer_trans_to_pandas(self.median_imputer \
                        .transform(data[self.median_imputer.feature_names_in_]),
                                                         index = data.index)
        fix_cols = [c for c in data.columns if c not in data_trans.columns]
        return pd.concat([data_trans, data[fix_cols]], axis=1)
    
    def set_scaled_features(self, data):
        """ Decide what kinds of scalings apply to which features.  For features
        thata are probably binary or categorical, do only min max scaling.  For
        features that are continuous (or have many levels), quantile scale
        before minmax.  
          Features are labeled as "probably binary or categorical" using the 
        threshold number of rows, a sample of data is used"""
        
        # Get level counts
        sample_n = np.min([data.shape[0], self.num_levels_scale_sample])
        levels_df = pd.concat([pd.DataFrame([data[c].sample(sample_n) \
                                            .value_counts().count()])  \
                              for c in self.features_in],
                             keys=self.features_in)
        
        self.features_minmax_scale = [c for c in data.columns if c in self.features_in] + \
            [c for c in self.median_imputer.get_feature_names_out() if c not in self.features_in]
        self.features_quantile_scale = list(levels_df[levels_df[0] > self.num_levels_scale] \
                                            .index.get_level_values(0))
    

        
    def quantile_scaler_create_fit(self, data_fill, transform = True):
        """Takes in median-filled data and creates/fits a quantile scaler,
        if applicable.  If there are no continuous predictor, no scaler is fit.
        Requires set_scaled_features() to have been called """
        
        # Ignore quantile scaling if no eligible features
        if (len(self.features_quantile_scale) == 0) & (transform):
            return data_fill
        elif (len(self.features_quantile_scale) == 0):
            return None
        
        # Create and fit the scaler
        self.quantile_scaler = QuantileTransformer(n_quantiles = self.quantile_levels)
        
        if transform:
            trans_data= self.quantile_scaler.fit_transform(data_fill[self.features_quantile_scale])
            return self.scaler_trans_append(data_fill, trans_data,
                                           scaler = self.quantile_scaler)
        else:
            self.quantile_scaler.fit(data_fill[self.features_quantile_scale])
            return None
    
    def quantile_scaler_transform(self, data):
        """Quantile scale data, using a fitted quantile scaler"""
        scaled_data= self.quantile_scaler.transform(data[self.features_quantile_scale])
        return self.scaler_trans_append(data, scaled_data, scaler=self.quantile_scaler)
    
    def minmax_scaler_create_fit(self, data_fill, transform = True):
        """Takes in median-filled data and creates/fits a minmax scaler,
        if applicable.  If there are no low-cardinality predictors, no scaler 
        is fit. Requires set_scaled_features() to have been called """
        
        # Ignore quantile scaling if no eligible features
        if (len(self.features_minmax_scale) == 0) & (transform):
            return data_fill
        elif (len(self.features_minmax_scale) == 0):
            return None
        
        # Create and fit the scaler
        self.minmax_scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
        
        if transform:
            trans_data= self.minmax_scaler.fit_transform(data_fill[self.features_minmax_scale])
            return self.scaler_trans_append(data_fill, trans_data,
                                           scaler=self.minmax_scaler)
        else:
            self.minmax_scaler.fit(data_fill[self.features_minmax_scale])
            return None
        
    def minmax_scaler_transform(self, data):
        """Quantile scale data, using a fitted quantile scaler"""
        scaled_data= self.minmax_scaler.transform(data[self.features_minmax_scale])
        return self.scaler_trans_append(data, scaled_data, scaler=self.minmax_scaler)
    

        
        
    def naics_encoder_create_fit(self, data, transform = True):
        """Label encode the NAICS feature. Add naics_reserve to OrdinalEncoder 
        to allow missing and/or unseen values """
        if self.naics_features is None:
            return data
        
        
        # Unknown value - needs padding
        miss_val = self.naics_missing - self.naics_reserve
        unk_val = self.naics_unknown - self.naics_reserve
        
        # Create and fit the encoder
        # Use negative missing/unknown values as we add naics_reserve later
        self.naics_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                             encoded_missing_value = miss_val,
                                             unknown_value = unk_val)  
        
        
        if transform:
            trans_data = self.naics_encoder.fit_transform(data[self.naics_features]) + \
                self.naics_reserve 
            #self.max_naics_cat = ([np.max(c) for c in self.naics_encoder.categories_])
            return self.scaler_trans_append(data, trans_data,
                                           scaler=self.naics_encoder)
        else:
            self.naics_encoder.fit(data[self.naics_features])
            #self.max_naics_cat = ([np.max(c) for c in self.naics_encoder.categories_])
            return None
                                            
    def naics_encoder_transform(self, data):
        """Transform NAICS data.  Add one to OrdinalEncoder to allow missing values """
        scaled_data= self.naics_encoder.transform(data[self.naics_features]) + \
            self.naics_reserve
        return self.scaler_trans_append(data, scaled_data, scaler=self.naics_encoder)
    
    def get_naics_encoder_levels(self):
        return len(self.naics_encoder.categories_[0])


    def fit_transform(self, data, transform = True):
        """Fit the imputer/scaler, and return transformed training data.  
        Median fills nulls and creates null indicator features.  
        Quantile transforms many-level fields, and min/max scales all 
        fields.  Input fields must be numeric and set during initialization."""
        
        # Create/fit the NAICS encoder
        trans_data1 = self.naics_encoder_create_fit(data, transform = transform)
        
        if self.features_in is None:
            return trans_data1
        
        # Create/fit median imputer for missing data, transform training data
        trans_data2 = self.median_imputer_create_fit(trans_data1, transform=transform)

        # Figure out which features to scale and not scale
        self.set_scaled_features(trans_data2)
        
        # Crete/fit the quantile scaler
        trans_data3 = self.quantile_scaler_create_fit(trans_data2, transform = transform)
        if not transform:
            trans_data3 = trans_data2
        
        # Create / fit the minmax scaler
        trans_data4 = self.minmax_scaler_create_fit(trans_data3, transform = transform)
        
        # Save the features after transform
        self.features_out = list(trans_data4.columns)
        
        return trans_data4

        
    def fit(self, data):
        """Fit the scalers, do not return transformed training data"""
        self.fit_transform(data, transform=False)
        
    def transform(self, data):
        """Transform dataset containing features, which will have null
        values median filled, and scaled. """
        data_1 = self.naics_encoder_transform(data)
        if self.features_in is None:
            return data_1
        data_2 = self.median_imputer_transform(data_1)
        data_3 = self.quantile_scaler_transform(data_2)
        
        return self.minmax_scaler_transform(data_3)
    
    def __init__(self, features = None, 
                 num_levels_scale = 5, 
                 num_levels_scale_sample = 100000,
                quantile_levels = 1000,
                naics_features = 'NAICS',
                reserve_naics_encoding_levels = 2,
                naics_missing = 0,
                naics_unknown = 1):
        """ Instantiates the custom scaler.  
          Inputs:
            features:  List of input features affected by the transformations.
              Other features in data passed to fit/transform functions (not including the
              NAICS feature).  Other features in the data will  be ignored.  If None, only 
              NAICS processing is done.
            num_levels_scale: If a feature contains this or fewer unique values,
              I use minmax scaling.  Above the threshold, quantile scaling is used.
            num_levels_scale_sample:  For efficiency on large data sets, a sample
              of cases is used to determine the number of levels per feature
              to determine whether the feature should be quantile or minmax scaled.
            quantile_levels: Number of quantiles to use for the quantile scaling
            naics_features: List of features (or single feature) for NAICS transform
            reserve_naics_encoding_levels: How many integer values to reseve in the 
              ordinal encoding of NAICS; these can be used to represent e.g. missing or
              unknown naics
            naics_missing: Value to be used for missing NAICS in the final dataset
            naics_unknown: Value to be used for unknown/unseen NAICS in the final dataset
              
        """
        
        # Predictors, except for NAICS
        if isinstance(features, list) or features is None:
            self.features_in = features
        else:
            self.features_in = [features]

        self.num_levels_scale = num_levels_scale
        self.num_levels_scale_sample = num_levels_scale_sample
        self.quantile_levels = quantile_levels
        self.naics_reserve = reserve_naics_encoding_levels
        self.naics_missing = naics_missing
        self.naics_unknown = naics_unknown
        
        if isinstance(naics_features, list) or naics_features is None:
            self.naics_features = naics_features
        else:
            self.naics_features = [naics_features]
        
        # During fit, I select some features for quantile scaling, others for
        # simple minmax scaling, based on level count, i.e. which are likely to
        # be categorical or binary.  
        features_quantile_scale = None
        features_minmax_scale = None
        
        # Output features may be more than input, set during fit()
        self.features_out = None
        
        # scikit-learn imputers/scalers to be created during fit() 
        median_imputer = None
        quantile_scaler = None
        minmax_scaler = None
        naics_encoder = None