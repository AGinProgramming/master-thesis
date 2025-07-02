'''
Title: Selected features for the NKI dataset based on dMRI dataset
Author: Annekoos Schaap, a.schaap@student.tue.nl
Date: 12/10/2022
Description:  Make a csv file with the selected features based on HyperparameterOptimization.py. Patients with multiple slices are averaged.
'''

import pandas as pd
import numpy as np

df = pd.read_csv(r'E:\ASchaap\ML\Stats\All_imaging_features_NKI.csv')

ADC_cols = [col for col in df.columns if 'ADC_v2' in col]
for col in df.columns:
    if 'ADC_v2' in col:
        df.rename(columns={col: 'ADC'+col[6:]}, inplace=True)

pnam = df['Patient name ' ]
X = df.drop(['Slice no. ', 'Label', 'Patient name '], axis=1)
y = df['Label']
labels = df[['Patient name ', 'Slice no. ', 'Label']]

f = open(r"E:\ASchaap\ML\features_HC_PCA_MRMR.txt", "r")
f_content = f.read()
selected_features = f_content.split(", ")
selected_features = selected_features[0:-1]
X = X[selected_features[0:62]]
df_new = X
df_new.insert(loc=0, column='Label', value=y)
df_new.insert(loc=0, column='Slice no. ', value=df['Slice no. '])
df_new.insert(loc=0, column='Patient name ', value=df['Patient name '])

# print(df_new)

# df_new = df

row09=df_new.index[df['Patient name ']=='MRI009'].tolist()
row12=df_new.index[df['Patient name ']=='MRI012'].tolist()
row14=df_new.index[df['Patient name ']=='MRI014'].tolist()
row23=df_new.index[df['Patient name ']=='MRI023'].tolist()
row31=df_new.index[df['Patient name ']=='MRI031'].tolist()

res = (df_new.groupby((df_new['Patient name '] != df_new['Patient name '].shift()).cumsum())
                  .mean()
                  .reset_index(drop=True))

patients = np.unique(df_new['Patient name '])

res.insert(loc=0, column='Patient name ', value=patients)

res.to_csv(r'E:\ASchaap\Feature extraction\Subset_imaging_features_NKI_kNN_62.csv')