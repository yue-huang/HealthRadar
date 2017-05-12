# Preprocessing: Select and modify features based on manual curation and missing data.
# Usage: python3 preprocessing.py LLCP2015-subset.csv CHCOCNCR cols-type-2015.csv

# Preprocessing steps:
# remove rows with NAs for outcome colname
# Remove cols to exclude and redundant cols
# Remove cols with NA > 90%
# Edit cols manually
# Replace 77,99 etc with NAs
# Replace NA with median (numerical, ordinal) or mode (categorical)

import sys
import csv
import pandas as pd
import numpy as np


def remove_NA_in_outcome_col(df, outcome_col_name):
    df = df[df[outcome_col_name].isin([1, 2])]
    return df


def remove_cols_to_exclude_or_redundant(df, colType):
    colRetain = colType[colType.exclude.isnull() & colType.redundant.isnull()].name
    df = df[(df.columns).intersection(colRetain)]
    return df


def manual_editing(df):
    if 'EMPLOY1' in df.columns:
        df.replace({'EMPLOY1': {9: np.nan}}, inplace=True)
    if 'VIDFCLT2' in df.columns:
        df.replace({'VIDFCLT2': {7: np.nan}}, inplace=True)
    if 'VIREDIF3' in df.columns:
        df.replace({'VIREDIF3': {7: np.nan}}, inplace=True)
    if 'VICTRCT4' in df.columns:
        df.replace({'VICTRCT4': {7: np.nan}}, inplace=True)
    if 'VIGLUMA2' in df.columns:
        df.replace({'VIGLUMA2': {7: np.nan}}, inplace=True)
    if 'VIMACDG2' in df.columns:
        df.replace({'VIMACDG2': {7: np.nan}}, inplace=True)
    if 'TETANUS' in df.columns:
        df.replace({'TETANUS': {2: 1}}, inplace=True)
        df.replace({'TETANUS': {3: 1}}, inplace=True)
    if 'HPVADVC2' in df.columns:
        df.replace({'HPVADVC2': {3: 2}}, inplace=True)
    if 'SCNTMNY1' in df.columns:
        df.replace({'SCNTMNY1': {8: np.nan}}, inplace=True)
    if 'SCNTMEL1' in df.columns:
        df.replace({'SCNTMEL1': {8: np.nan}}, inplace=True)
    if 'SCNTWRK1' in df.columns:
        df.replace({'SCNTWRK1': {98: 0}}, inplace=True)
        df.replace({'SCNTWRK1': {97: np.nan}}, inplace=True)
    if 'SCNTLWK1' in df.columns:
        df.replace({'SCNTLWK1': {98: 0}}, inplace=True)
        df.replace({'SCNTLWK1': {97: np.nan}}, inplace=True)
    if '_RACE' in df.columns:
        df.replace({'_RACE': {9: np.nan}}, inplace=True)
    if 'DROCDY3_' in df.columns:
        df.replace({'DROCDY3_': {900: np.nan}}, inplace=True)
    if '_DRNKWEK' in df.columns:
        df.replace({'_DRNKWEK': {99900: np.nan}}, inplace=True)
    if 'STRFREQ_' in df.columns:
        df.replace({'STRFREQ_': {99000: np.nan}}, inplace=True)
    return df


valsToReplace = {
    9: [7, 9],
    99: [77, 88, 99],
    999: [777, 999],
    9999: [7777, 9999],
    999999: [777777, 999999],
}


def replace_meaningless_vals(dataframe, valsToReplace):
    max = dataframe.max(axis=0)
    newdf = pd.DataFrame(dataframe)
    for colname in newdf.columns:
        if colname not in max:
            # print("Dropping column", colname)
            newdf.drop(colname, axis=1, inplace=True)
        elif max[colname] in valsToReplace:
            for val in valsToReplace[max[colname]]:
                newdf.replace({
                    colname: {
                        val: np.nan
                    }
                }, inplace=True)
    return newdf


def impute_NA(df, colType):
    for col in df.columns:
        if df[col].isnull().any():
            type = colType[(colType.name == col)].type.iloc[0]
            if type == 'n' or type == 'o':
                df[col].fillna(value=df[col].median(), inplace=True)
            elif type == 'c':
                df[col].fillna(value=df[col].mode().iloc[0], inplace=True)
    return df


def main(data_csv_name, outcome_col_name, coltype_csv_name):
    stats = {}
    df = pd.read_csv(data_csv_name, sep=',')
    colType = pd.read_csv(coltype_csv_name, sep=',')
    stats['initial'] = [df.shape[0], df.shape[1]]

    df = remove_NA_in_outcome_col(df, outcome_col_name)
    stats['remove_NA_in_outcome_col'] = [df.shape[0], df.shape[1]]

    df = remove_cols_to_exclude_or_redundant(df, colType)
    stats['remove_cols_to_exclude_or_redundant'] = [df.shape[0], df.shape[1]]

    df.dropna(axis=1, thresh=0.1 * df.shape[0], inplace=True)
    stats['dropNA>90%_1st'] = [df.shape[0], df.shape[1]]

    df = manual_editing(df)
    stats['manual_editing'] = [df.shape[0], df.shape[1]]

    df = replace_meaningless_vals(df, valsToReplace)
    stats['replace_meaningless_vals'] = [df.shape[0], df.shape[1]]

    df.dropna(axis=1, thresh=0.1 * df.shape[0], inplace=True)
    stats['dropNA>90%_2nd'] = [df.shape[0], df.shape[1]]

    df = impute_NA(df, colType)
    stats['impute_NA'] = [df.shape[0], df.shape[1]]

    with open('stats_preprocessing.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['procedure','row_num','col_num'])
        for key, value in stats.items():
            print(key, value)
            writer.writerow([key] + value)


    df.to_csv(sys.argv[1] + '_missingDataHandling.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
