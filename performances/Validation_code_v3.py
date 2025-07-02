'''
Title: Test model trained on dMRI dataset on NKI dataset
Author: Annekoos Schaap, a.schaap@student.tue.nl
Date: 25/10/2022
Description: 
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from split import RepeatedStratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, StratifiedGroupKFold, cross_validate
from sklearn.feature_selection import RFE
import pickle
from sklearn.inspection import permutation_importance as pfi

sel_method = 'rfe'
with open('best_models_'+ sel_method + '.pickle', 'rb') as f:
    best_models = pickle.load(f)
with open('feat_lists_'+ sel_method + '.pickle', 'rb') as f:
    feat_list = pickle.load(f)

# Load dMRI dataset (train)
df_dMRI = pd.read_csv(r'Z:\GENIM\Annekoos Schaap - code\ML\Stats\data_allSequences_dMRI_without_overlap_v4.csv')
pnam_dMRI = df_dMRI['Patient name ' ]
X_dMRI = df_dMRI.drop(['Slice no. ', 'Label', 'Patient name '], axis=1)
y_dMRI = df_dMRI['Label']

# Load NKI dataset (test)
df_NKI = pd.read_csv(r'Z:\GENIM\Annekoos Schaap - code\ML\Stats\All_imaging_features_NKI.csv')
X_NKI = df_NKI.drop(['Slice no. ', 'Label', 'Patient name '], axis=1)
y_NKI = df_NKI['Label']

# Load the subset of features
f = open(r"C:\projects\genim\server_code\features_HC_PCA_MRMR.txt", "r")
f_content = f.read()
selected_features = f_content.split(", ")
selected_features = selected_features[0:-1]

specificity = make_scorer(recall_score, pos_label=0)

multi_scorings = {"AUC": "roc_auc", "ACC": "accuracy", "bACC" :'balanced_accuracy', 
                  "SENS": "recall", "SPEC": specificity}

# Select classifier types with previously tuned hyperparameters

sc = StandardScaler()
metrics = ['AUC', 'bACC', 'ACC']
pfis = {}

for mtr in metrics:
    pfis_per_model = {}
    print(f"*************Optimizing {mtr} *********")
    for mdl in best_models[mtr].keys():
        print('%%%%% Validation of: ', mdl )
        model = best_models[mtr][mdl]
        N_o = model.n_features_in_
        # if 'gamma' in model.__dict__.keys():
        #     N_o = 34
        print(f'Selected features:', N_o)
        if sel_method == 'mrmr':
            # Select subset of features
            X_dMRI_sub = X_dMRI[selected_features[0:N_o]]
        elif sel_method == 'rfe':
            sel_cols = feat_list[mtr][mdl]
            X_dMRI_sub = X_dMRI[sel_cols]
        
        # Train model on dMRI dataset
        
        X_dMRI_sub = sc.fit_transform(X_dMRI_sub)
        cv = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats = 20, random_state=42)
        
        scores = cross_validate(model, X_dMRI_sub, y_dMRI, groups=pnam_dMRI, scoring = multi_scorings, cv = cv, n_jobs=-1)
        
        print("Performance over training data:")
        for metric in multi_scorings.keys():
            print("%s: %.3f (%.3f)" % (metric, np.mean(scores['test_'+ metric]), np.std(scores['test_'+metric])))

        # print('AUC on training data: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        if sel_method == 'mrmr':
            # Select subset of features
            X_NKI_sel = X_NKI[selected_features[0:N_o]]
        elif sel_method == 'rfe':
            sel_cols = feat_list[mtr][mdl] 
            print(len(sel_cols))
            X_NKI_sel = X_NKI[sel_cols]
        
        X_test = sc.fit_transform(X_NKI_sel)
        y_test = y_NKI
        model.fit(X_dMRI_sub, y_dMRI)
        y_predict = model.predict(X_test)
        acc = accuracy_score(y_test, y_predict)
        bacc = balanced_accuracy_score(y_test, y_predict)
        sens = recall_score(y_test, y_predict) # Sensitivity is the recall of the postive class
        spec = recall_score(y_test, y_predict, pos_label=0) # Specificity is the recall of the negative class
        auc = roc_auc_score(y_test, y_predict)
        print("Performance over testing data:\n ACC: %.3f bACC: %.3f SENS: %.3f SPEC: %.3f AUC: %.3f" % (acc, bacc, sens, spec, auc))
        #PFI on test data
        res_pfi = pfi(model, X_test, y_test,
                                    scoring=multi_scorings[mtr],
                                    n_repeats=10,
                                    random_state=42)
                
        pfis_per_model[mdl]= res_pfi
    pfis[mtr] = pfis_per_model
    
with open('pfis_validation_' + sel_method + '.pickle', 'wb') as f:
    pickle.dump(pfis, f)
        # save_df_nam = clf_type + '_results.csv'
        # PFI_df.to_csv(os.path.join(r'E:\ASchaap\Results_ML\PFI', save_df_nam),index=False)
        # PFI_df2.to_csv(os.path.join(r'E:\ASchaap\Results_ML\PFI', save_df_nam2),index=False)
        
