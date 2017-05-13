# Transformation: Scale numerical data, convert categorical data to binary data.
# Usage: python3 transformation.py LLCP2015.csv_preprocessing.csv CHCOCNCR cols-type-2015.csv

# Transformation steps:
# convert outcome column to binary data
# convert categorical data to multiple columns with binary data
# Scale numerical data
# Split data to three parts: training, cv, test sets.

import pandas as pd
import sys
from random import sample
import csv


def separate_cols(df, outcome_col, colType):
    y = df[outcome_col]
    x = df.drop(outcome_col, axis=1)
    x_numerical_ordinal = pd.DataFrame()
    x_categorical = pd.DataFrame()
    for col in x.columns:
        type = colType[(colType.name == col)].type.iloc[0]
        if type == 'n' or type == 'o':
            x_numerical_ordinal[col] = x[col]
        elif type == 'c':
            x_categorical[col] = x[col]
    return y, x_numerical_ordinal, x_categorical


def transformation(y, x_numerical_ordinal, x_categorical):
    y.replace(to_replace=2, value=0, inplace=True)
    for col in x_numerical_ordinal.columns:
        min = x_numerical_ordinal[col].min()
        max = x_numerical_ordinal[col].max()
        if min == max:
            x_numerical_ordinal.drop(col, axis=1, inplace=True)
        else:
            x_numerical_ordinal[col] = (x_numerical_ordinal[col] - min) / (max - min)

    x_categorical = pd.get_dummies(x_categorical, columns=x_categorical.columns, drop_first=True)
    stats['x_numerical_ordinal_transformed'] = [x_numerical_ordinal.shape[0], x_numerical_ordinal.shape[1]]
    stats['x_categorical_transformed'] = [x_categorical.shape[0], x_categorical.shape[1]]
    df_transformed = pd.concat([y, x_numerical_ordinal, x_categorical], axis=1)
    return (df_transformed)


def split_data(df, train_percentage):
    row_num = df.shape[0]
    split_num = round(row_num * train_percentage)
    random = sample(range(row_num), k=len(range(row_num)))
    df_train = df.iloc[random[0:split_num]]
    df_test = df.iloc[random[split_num:]]
    return df_train, df_test


def main(data_csv_name, outcome_col_name, coltype_csv_name):
    print("Reading in files...")
    df = pd.read_csv(data_csv_name, sep=',')
    colType = pd.read_csv(coltype_csv_name, sep=',')
    stats['initial'] = [df.shape[0], df.shape[1]]

    print('Separating outcome, numerical/ordinal and categorical columns...')
    y, x_numerical_ordinal, x_categorical = separate_cols(df, outcome_col_name, colType)
    stats['x_numerical_ordinal'] = [x_numerical_ordinal.shape[0], x_numerical_ordinal.shape[1]]
    stats['x_categorical'] = [x_categorical.shape[0], x_categorical.shape[1]]

    print('Transforming columns...')
    df_transformed = transformation(y, x_numerical_ordinal, x_categorical)
    stats['transformed'] = [df_transformed.shape[0], df_transformed.shape[1]]
    df_transformed.to_csv(sys.argv[1] + '_transformed.csv', index=False)

    print('Splitting data...')
    df_train, df_test = split_data(df_transformed, 0.8)

    stats['training_set'] = [df_train.shape[0], df_train.shape[1]]
    stats['test_set'] = [df_test.shape[0], df_test.shape[1]]

    print('writing files...')
    with open('stats_transformation.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['procedure', 'row_num', 'col_num'])
        for key, value in stats.items():
            writer.writerow([key] + value)

    df_train.to_csv(sys.argv[1] + '_transformed_train.csv', index=False)
    df_test.to_csv(sys.argv[1] + '_transformed_test.csv', index=False)


if __name__ == '__main__':
    stats = {}
    main(sys.argv[1], sys.argv[2], sys.argv[3])
