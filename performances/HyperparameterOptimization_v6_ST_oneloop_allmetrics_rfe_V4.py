'''
Title: Finding optimal number of features with PCA and MRMR without down- or upsampling
Author: Annekoos Schaap, a.schaap@student.tue.nl
Date: 12/10/2022
Description:
1. Define dataset
(2. Remove highly correlated features) --> Irrelevant due to PCA
3. Apply PCA to find number of relevant features
4. Use minimum-redundancy maximum-relevance to take the most relevant features
5. Perform hyperparameter tuning using a random search approach
6. Find the optimum number of features
7. Optimize hyperparameters with the features selected in the step above.
'''

from datetime import datetime
import pandas as pd
from scipy.stats import loguniform 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
from calcDrop_v2 import corrX_new
import numpy as np
from sklearn.inspection import permutation_importance as pfi
from mrmr import mrmr_classif
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, GroupShuffleSplit, cross_validate
from split import RepeatedStratifiedGroupKFold
from imblearn.metrics import specificity_score
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFE

#from sklearn.metrics import make_scorer Define dataset
# Drop highly correlated variables from dataset (218, 1267) --> (218, 609)
df = pd.read_table(r'Z:\GENIM\Annekoos Schaap - code\ML\Stats\data_allSequences_dMRI_without_overlap_v4.csv')
pnam = df['Patient name ' ]
X = df.drop(['Slice no. ', 'Label', 'Patient name '], axis=1)
y = df['Label']
to_drop = corrX_new(X, y, cut = 0.9)
X = X.drop(to_drop, axis=1)
df = df.drop(to_drop, axis=1)
print("Number of features left after dropping highly correlated features: ", np.shape(X)[1])
# df.to_csv(r'E:\ASchaap\ML\Stats\data_allSequences_dMRI_without_overlap_after_drop_v4.csv')

# df = pd.read_csv(r'E:\ASchaap\ML\Stats\data_allSequences_dMRI_without_overlap_after_drop_v4.csv')
# df.drop(['Unnamed: 0'], axis=1, inplace=True)
# X = df.drop(['Slice no. ', 'Label', 'Patient name '], axis =1)
# pname = df['Patient name ']
# y = df['Label']
# print("Number of features left after dropping highly correlated features: ", np.shape(X)[1])

# Apply PCA to retain 95% variance. StandardScaler to standardize data. (218, 609) --> (218, 122)

# from sklearn.decomposition import PCA

scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
# pca = PCA(.95).fit(X_scaled)
# print("Number of components left after PCA: ", pca.n_components_)
# features_PCA = pca.transform(X_scaled)

# MRMR ranking or univariate feature selection and picking top 20

# 

# X = df.drop(['Slice no. ', 'Label', 'Patient name '], axis =1)
# selected_features = mrmr_classif(X=X, y=y, K=pca.n_components_)
# X = X[selected_features]
# f = open(r"E:\ASchaap\ML\features_HC_PCA_MRMR.txt", "w+")
# for feature in selected_features:
#     f.write(feature + ", ")
# f.close()
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
print(dt_string)
f = open(r"C:\projects\genim\server_code\features_HC_PCA_MRMR.txt", "r")
f_content = f.read()
selected_features = f_content.split(", ")
X = X[selected_features[0:-1]]
featNames = X.columns
# These features should be scaled before using them to train the models

# Coarse grid search
# Define models and parameters for random search


models = ["kNN","SVM","LR"]

# Define search


cv_outer = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=10, random_state=42) # Split less (split = 5), and repeat more (10)
multi_scorings = {"AUC": "roc_auc", "ACC": "accuracy", "bACC" :'balanced_accuracy', 
                  "SENS": "recall", "SPEC": make_scorer(specificity_score)}

#cross-validation
#from imblearn.over_sampling import SMOTE

# Define SMOTE for upsampling the imbalanced dataset
# sm = SMOTE(random_state=42)
scoring_metrics=['AUC','bACC', 'ACC', 'SENS', 'SPEC']
for scoring_metric in scoring_metrics:

    for model_name in models:
        pfis = []
        final_dict, final_dict_avg = {}, {}
        result_outer = list()
        features_number, grid_score, grid_parms = [], [] , []
        n_features_list, result_outer_mean, result_outer_std = [], [], []
        df_res_cv = pd.DataFrame()
        rankings,selecfeat_names  = [],[]
        best_models = []
        
        if model_name == "LR":
            print("Evaluating for {}".format(model_name))
            # Define the model
            model = LogisticRegression(class_weight='balanced')
    
            # Define the search space
            space = dict()
            space['solver'] = ['liblinear'] # liblinear is most logical for the size of the dataset (small) and the task (binary)
            space['penalty'] = ['l1', 'l2']
            space['C'] = loguniform(1e-5, 10)  
            n_iter = 50
    
        elif model_name == "kNN":
            print("Evaluating for {}".format(model_name))
            model = KNeighborsClassifier()
            space = dict()
            space['n_neighbors'] = list(range(3, 30))
            space['weights'] = ['uniform', 'distance']
            space['metric'] = ['minkowski']
            n_iter = 20
    
        elif model_name == "SVM":
            print("Evaluating for {}".format(model_name))
            model = SVC(class_weight='balanced')
            space = dict()
            space['kernel'] = ['poly','rbf', 'sigmoid']
            space['C'] = loguniform(1e-4, 10)
            space['gamma'] = loguniform(1e-4, 10)
            n_iter = 100
            
        
        for n_features in range(10, 64): #range(5, np.shape(X)[1]+1): # in [5, 10, 20, 50, 100]: #
            # Select feature subset
            print("Using %s features" % (n_features))
            Xscaled = scaler.fit_transform(X, y)
            
            selector = RFE(LogisticRegression(class_weight='balanced', 
                                         C=0.034, penalty = 'l2',
                                         solver = 'liblinear'),
                                         n_features_to_select=n_features)
            selector.fit(Xscaled, y)
            X_sub = X.iloc[:,selector.support_]
            selecfeat_names.append(featNames[selector.support_]) 
            rankings.append(selector.ranking_)


            cv_inner = list(RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=20, random_state=42).split(X_sub, y, pnam))
            # Standardize data
            X_sub = scaler.fit_transform(X_sub, y)
    
    
            # Define search
            random_search = RandomizedSearchCV(model, space, n_iter=n_iter, scoring=multi_scorings[scoring_metric], 
                                               n_jobs=-1, cv=cv_inner, error_score=0, refit=True)
    
            
            
            # # Apply SMOTE for imbalanced dataset
            # X_train, y_train = sm.fit_resample(X_train, y_train)
    
            # Execute search
            result_inner = random_search.fit(X_sub, y)
    
            # Get the best performing model fit on the whole training set
            best_model = result_inner.best_estimator_
            print(best_model)
            best_models.append(best_model)
            cv_validate = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=20, random_state=42)
            res_all = cross_validate(best_model, X_sub, y, groups=pnam, scoring = multi_scorings, cv = cv_validate, n_jobs=-1)
            # Store the result
        
            scores = res_all["test_"+scoring_metric]
           
            res_all_df = pd.DataFrame.from_dict(res_all)
            res_all_df["n_features"] = np.ones((len(scores),1))*n_features
            df_res_cv = df_res_cv.append(res_all_df)
            
            #Calculate and save the PFI
            
            res_pfi = pfi(best_model, X_sub, y,
                                        scoring=multi_scorings[scoring_metric],
                                        n_repeats=10,
                                        random_state=42)
                    
            pfis.append(res_pfi)
            
            
            #Extract interesting values for progress
            print('Model: %s N_feat: %.3f %s: %.3f (%.3f)' 
                  % (model_name, n_features, scoring_metric,np.mean(scores), np.std(scores)))
            n_features_list.append(n_features)
            result_outer_mean.append(np.mean(scores))
            result_outer_std.append(np.std(scores))
    
        #Store results
        # final_dict = {'n features': features_number, 'score': grid_score, 'parms': grid_parms}
        final_dict_avg ={'n features': n_features_list, 'mean score': result_outer_mean, 'std score': result_outer_std}
        df_res_cv.to_csv(r'C:\projects\genim\server_code\results\{}_oneloop_rfe_{}_{}.csv'
                         .format(model_name, scoring_metric, dt_string))
        final_df_avg = pd.DataFrame(final_dict_avg)
        final_df_avg.to_csv(r'C:\projects\genim\server_code\results\{}_oneloop_rfe_{}_{}_avg.csv'
                            .format(model_name, scoring_metric, dt_string))
        
        with open(r'C:\projects\genim\server_code\results\{}_oneloop_{}_{}_rfe_features_selection.pickle'
                  .format(model_name, scoring_metric, dt_string), 'wb') as f:
            pickle.dump([pfis,selecfeat_names, rankings], f)

        with open(r'C:\projects\genim\server_code\results\{}_oneloop_{}_{}_rfe_hparams.pickle'
                          .format(model_name, scoring_metric, dt_string), 'wb') as f:
            pickle.dump(best_models, f)
            
    


