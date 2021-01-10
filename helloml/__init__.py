import pandas as pd
import numpy as np
from pathlib import Path
from pandas_profiling import ProfileReport

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class HelloDataset():
    '''
    Stores methods for dataset operations.
    '''

    # Dataloading
    def load(self, path_obj):
        '''
        Loads a .csv into a class attribute.
        param path_obj: Path object pointing to dataset stored as .csv
        '''
        print('Loading data...')
        self.df = pd.read_csv(path_obj)
        print('Data loaded.')
        # print('Columns found:')
        # for col in self.df.columns:
        #     print(f'Feature name: {col}\tDatatype: {self.df[col]}\tMissing values: \
        #         {self.df[col].isna().count()}')

    def explore(self):
        self.profile = ProfileReport(
            self.df, title='Explore the Dataset', minimal=True, progress_bar=False)
        return self.profile.to_notebook_iframe()

    # Data cleaning
    def drop_feature(self, feature_list):
        '''
        Removes the given column(s).
        '''
        self.df = self.df.drop(feature_list, axis=1)
        print(f'Feature(s) {feature_list} removed.')

    def convert_to_numerical(self, feature_list):
        '''
        Converts the given column(s) to dummies.
        '''
        print(
            f'Converting features {feature_list} to numerical by creating dummy variables...')
        dummy_cols = []
        for col in feature_list:
            dummies = pd.get_dummies(
                self.df[col], drop_first=True, prefix=col)
            dummy_cols.extend(dummies.columns)
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df = self.df.drop(col, axis=1)
        print(f'Features removed: {feature_list}.')
        print(f'Dummy features added: {list(dummy_cols)}')

    # def drop_example(self, row):
    #     '''
    #     Removes the given row. 
    #     Commented out due to need for better data exploration support for 
    #     appropriate usage.
    #     '''
    #     self.df = self.df.drop(row, axis=0)
    #     print(f'Example {row} removed.')

    # Feature engineering
    def sum_feature(self, new_feat, feature_list):
        '''
        Sum columns.
        param feature_list: List of column names (as strings) to sum with each other.
        param new_feat: The name of the new column.
        '''
        self.df[new_feat] = self.df[feature_list].sum(axis=1)
        print(
            f'Created new feature "{new_feat}" by summing features {feature_list}.')

    def subtract_feature(self, new_feat, feat_a, feat_b):
        '''
        Divide one column by another.
        param feat_a: The column from which to subtract.
        param feat_b: The column to subtract from feat_a.
        param new_feat: The name of the new column.
        '''
        self.df[new_feat] = self.df[feat_a] - self.df[feat_b]
        print(
            f'Created new feature "{new_feat}" by dividing {feat_num} by {feat_denom}.')

    def divide_feature(self, new_feat, feat_num, feat_denom):
        '''
        Divide one column by another.
        param feat_num: The column containing the feature in the numerator.
        param feat_denom: The column containing the feature in the denominator.
        param new_feat: The name of the new column.
        '''
        self.df[new_feat] = self.df[feat_num]/self.df[feat_denom]
        print(
            f'Created new feature "{new_feat}" by dividing {feat_num} by {feat_denom}.')

    def multiply_feature(self, new_feat, feature_list):
        '''
        Multiply one column by another.
        param feature_list: List of column names (as strings) to multiply by each other.
        param new_feat: The name of the new column.
        '''
        self.df[new_feat] = self.df[feature_list].prod(
            axis=1, numeric_only=True)
        print(
            f'Created new feature "{new_feat}" by multiplying features {feature_list}.')

    def set_target_feature(self, target_feature, scale=True):
        '''
        param target_feature: name of column to predict.
        All other columns will be normalised and used for prediction.
        '''
        self.X = self.df.drop(target_feature, axis=1)
        self.Y = self.df[target_feature]
        
        print(f'Target feature set to {target_feature}.')
        print(f'Features used for prediction:')
        for i in list(self.X.columns):
            print(f'- {i}')

        if scale:
            scaler = RobustScaler()
            num_df = scaler.fit_transform(self.X.select_dtypes(include=[np.number]))
            self.X.update(num_df)

    def impute_missing_values(self, feature_list, imputation_type='auto'):
        '''
        Impute missing values in the selected columns of the dataset
        param feature_list: List of column names (as strings) to impute.
        param imputation_type: What kind of imputation to use (auto, mode, or mean)
        '''
        
        for feature in feature_list:
            num_na = sum(self.df[feature].isna())
            if num_na == 0:
                print(f'No null values detected in column "{feature}"... skipping to next feature')
                continue
            elif imputation_type == 'auto':
                if (self.df[feature].dtype == 'O' or self.df[feature].dtype == 'str'):
                    print(f'{num_na} null values detected in column "{feature}"... replacing with the mode')
                    new_feature = self.df[feature].fillna(self.df[feature].mode()[0])
                else:
                    print(f'{num_na} null values detected in column "{feature}"... replacing with mean')
                    new_feature = self.df[feature].fillna(self.df[feature].mean()[0])
            elif imputation_type == 'mode':
                print(f'{num_na} null values detected in column "{feature}"... replacing with the mode')
                new_feature = self.df[feature].fillna(self.df[feature].mode()[0])
            elif imputation_type == 'mean' and (self.df[feature].dtype == 'O' or self.df[feature].dtype == 'str'):
                print(f'Error: Cannot impute mean of a non-numerical feature. Skipping imputation of feature "{feature}"...')
                continue
            elif imputation_type == 'mean':
                print(f'{num_na} null values detected in column "{feature}"... replacing with mean')
                new_feature = self.df[feature].fillna(self.df[feature].mean()[0])
            else:
                raise AssertionError('Error: Incorrect imputation type. Please enter one of the options: "auto", "mean", or "mode" only.')
            self.df[feature] = new_feature                

# ML models
class HelloModel():
    names = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbours']
    obj_dict = dict(zip(names, [
        LogisticRegression(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
    ]))
    req_dict = dict(zip(names, [
        ['Requires numerical features',],
        ['Requires numerical features', 'Cannot process missing values'],
        ['Requires numerical features'],
    ]))
    params_dict = dict(zip(names, [
        ['None'],
        ['None'],
        ['No. of neighbours to consider: before training, run the set_neighbours method to inform the model how many neighbours to consider. Note that test performance is typically worse when this number is less than 2']
    ]))

    def __init__(self, model_name):
        self.name = model_name
        self.model = self.obj_dict[model_name]
        self.requirements = self.req_dict[model_name]
        self.params = self.params_dict[model_name]
        
        print(f'Model {self.name} initialised.')
        print(f'Requirements for this model:')
        for i, req in enumerate(self.requirements):
            print(f'{i+1}. {req}.')
        print('\n')
        print(f'Parameters to tune for this model:')
        for i, param in enumerate(self.params):
            print(f'{i+1}. {param}.')

    def train(self, hellodata):
        print(f'Training model {self.name}...')
        self.model.fit(hellodata.X, hellodata.Y)
        print('Done.')

    def test(self, hellodata):
        score = self.model.score(hellodata.X, hellodata.Y)
        print(f'Accuracy: {round(score*100, 1)}% of your predictions were correct.')

    def set_neighbours(self, n_neighbours):
        '''
        Set the number of neighbours to consider in the K-Nearest Neighbours model.
        param n_neighbours: integer to use as number of neighbours
        '''
        assert isinstance(self.model, KNeighborsClassifier), 'In order to set the number of neighbours, you must use K-Nearest Neighbours as your model.'
        self.model = KNeighborsClassifier(n_neighbours)
        print(f'Number of neighbours in {self.name} set to {n_neighbours}.')