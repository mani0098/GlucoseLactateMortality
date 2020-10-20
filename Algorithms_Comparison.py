import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
# from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
# import seaborn as sns

ML_prim = pd.read_csv('ML_ready.csv')
ML_analysis = copy.deepcopy(ML_prim)
ML_analysis = ML_analysis.drop(['Unnamed: 0', 'uniquepid'], axis=1)

X = ML_analysis.values[:,0:11]
Y = ML_analysis.values[:,11]
# smote = SMOTE(sampling_strategy='minority', k_neighbors=1)
# X_sm, y_sm = smote.fit_sample(X, Y)

# X = X_sm
# Y = y_sm
Y = np.where(Y == 'Alive', 0, 1)

# split data into train and test sets
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='newton-cg', penalty='l2', class_weight='balanced', random_state=42)))
models.append(('LDA', LinearDiscriminantAnalysis(n_components=1, store_covariance=True, priors=[0.5, 0.5], solver='svd', shrinkage=None)))
models.append(('KNN', KNeighborsClassifier(n_neighbors=25, p=2, weights='distance', metric='euclidean', algorithm='auto', n_jobs=-1)))
models.append(('CART', XGBClassifier(learning_rate=0.4, max_depth=128, n_estimators=1)))
models.append(('NB', GaussianNB(priors=[0.5, 0.5])))
models.append(('XGB', XGBClassifier(learning_rate=0.2, max_depth=64, n_estimators=100)))
models.append(('SVM', SVC(gamma='scale', probability=True, class_weight='balanced', shrinking=True, random_state=0)))
# evaluate each model in turn
results = []
names = []
for name, model in models:
 	kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)#StratifiedKFold(n_splits=5, random_state=42, shuffle=True)#
 	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
 	results.append(cv_results)
 	names.append(name)
 	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
# Compare Algorithms
plt.boxplot(results, labels=names)
# sns.boxplot(data=results)
plt.title('Algorithms Comparison')
plt.ylabel('AUROC')
# plt.xticks(range(len(names)), names, ha = 'center', rotation = 'horizontal')
plt.show()