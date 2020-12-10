import numpy as np
import pandas as pd
from pathlib import Path



class Dataset():
    '''
    Stores methods for dataset operations.
    '''

    # Dataloading
    def load(path_obj):
        '''
        Loads a .csv into a class attribute.
        param path_obj: Path object pointing to dataset stored as .csv
        '''
        print('Loading data...')
        self.df = pd.read_csv(path_obj)
        print('Data loaded.')
        print('Columns found:')
        for col in self.df.columns:
            print(f'Feature name: {col}\tDatatype: {self.df[col]}\tMissing values: \
                {self.df[col].isna().count()}')

    # Data cleaning
    def impute_feature(col):
        '''
        Replaces missing values with the mean for a given column.
        param col: column to replace missing values
        '''
        self.df = self.df.fillna(col, self.df[col].mean())
        print('Missing values for feature {col} have been replaced with the \
            mean of its non-missing values.')


    def drop_feature(col):
        '''
        Removes the given column.
        '''
        self.df = self.df.drop(col, axis=1)
        print(f'Feature {col} removed.')


    def drop_example(row):
        '''
        Removes the given row.
        '''
        self.df = self.df.drop(row, axis=0)
        print(f'Example {row} removed.')

    # Feature engineering
    def divide(cola, colb, newcol):
        '''
        Divide one column by another.
        param cola: The column containing the feature in the numerator.
        param colb: The column containing the feature in the denominator.
        param newcol: The name of the new column.
        '''
        self.df[newcol] = self.df[cola]/self.df[colb]

    def multiply(cola, colb, newcol):
        '''
        Multiply one column by another.
        param cola: The column containing the first feature to be multiplied.
        param colb: TThe column containing the second feature to be multiplied.
        param newcol: The name of the new column.
        '''
        self.df[newcol] = self.df[cola]*self.df[colb]
