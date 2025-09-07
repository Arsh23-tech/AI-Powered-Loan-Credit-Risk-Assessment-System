import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Removes outliers from a column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def engineer_features(df):
    """
    Adds aggregated features based on PAY_*, BILL_AMT*, PAY_AMT* series.
    """
    # Aggregated PAY_* features
    pay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    df['avg_pay_status'] = df[pay_cols].mean(axis=1)
    df['std_pay_status'] = df[pay_cols].std(axis=1)
    df['max_pay_status'] = df[pay_cols].max(axis=1)

    # Aggregated BILL_AMT* features
    bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    df['std_bill_amt'] = df[bill_cols].std(axis=1)
    df['max_bill_amt'] = df[bill_cols].max(axis=1)

    # Aggregated PAY_AMT* features
    pay_amt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    df['avg_pay_amt'] = df[pay_amt_cols].mean(axis=1)
    df['std_pay_amt'] = df[pay_amt_cols].std(axis=1)

    return df

def clean_and_engineer(df):
    """
    Main preprocessing function that:
    - Removes outliers from LIMIT_BAL and AGE
    - Applies log transform to LIMIT_BAL
    - Engineers new aggregate features
    """
    # Step 1: Remove outliers
    df = remove_outliers_iqr(df, 'LIMIT_BAL')
    df = remove_outliers_iqr(df, 'AGE')

    # Step 2: Log-transform LIMIT_BAL
    df['log_limit_bal'] = np.log1p(df['LIMIT_BAL'])

    # Step 3: Add aggregated features
    df = engineer_features(df)

    return df