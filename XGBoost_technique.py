import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE

# roc curve and auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# from matplotlib.legend import Legend
import seaborn as sns

##############################################################################################################################
ML_prim = pd.read_csv('ML_ready.csv')
sns.scatterplot(x=ML_prim['minimumglucose'], y=ML_prim['minimumlactate'], style=ML_prim['mortality'], hue=ML_prim['mortality'], hue_order = ['Expired', 'Alive'])
# sns.scatterplot(x=ML_prim['meanglucose'], y=ML_prim['meanlactate'], style=ML_prim['mortality'], hue=ML_prim['mortality'], hue_order = ['Expired', 'Alive'])
# sns.scatterplot(x=ML_prim['maximumglucose'], y=ML_prim['maximumlactate'], style=ML_prim['mortality'], hue=ML_prim['mortality'], hue_order = ['Expired', 'Alive'])
plt.xlim(xmax = 3000)
plt.ylim(ymax = 700)
plt.show()

ML_prim2 = ML_prim[['age', 'gender', 'admissionweight', 'dischargeweight', 'minimumglucose', 'meanglucose', 'maximumglucose', 'minimumlactate', 'meanlactate', 'maximumlactate', 'diabetes', 'mortality']]
sns.pairplot(ML_prim2, hue='mortality', hue_order = ['Expired', 'Alive'])
##############################################################################################################################

ML_prim = pd.read_csv('ML_ready.csv')
ML_analysis = copy.deepcopy(ML_prim)
ML_analysis = ML_analysis.drop(['Unnamed: 0', 'uniquepid'], axis=1)
# shape
print(ML_analysis.shape)

# descriptions
print(ML_analysis.describe())

# class distribution
print(ML_analysis.groupby('mortality').size())

# Data to plot
labels = ['Expired', 'Alive']
sizes = [len(ML_analysis[ML_analysis['mortality'] == 'Expired']), len(ML_analysis[ML_analysis['mortality'] == 'Alive'])]
colors = ['red', 'yellowgreen']
explode = (0.2, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.title('Expired vs. Alive Patients Ratio')
plt.show()

ML_analysis.corr()
# mask = np.triu(np.ones_like(ML_analysis.corr())).astype(np.bool)
f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(ML_analysis.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

# split data into X and y
X = ML_analysis.values[:,0:11]
Y = ML_analysis.values[:,11]

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(frameon=False, loc='center left')
    plt.show()
    
plot_2d_space(X, Y, 'Imbalanced Data')


from collections import Counter
counter = Counter(Y)
# print(counter)
# for label, _ in counter.items():
# 	row_ix = where(Y == label)[0]
# 	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
# pyplot.legend(frameon=False, loc='center left')
# pyplot.show()

labels = ['Alive', 'Expired']
fig = plt.figure()
ax = fig.add_subplot(111)
pd.value_counts(Y).plot.bar()
plt.title('Mortality Histogram')
plt.xlabel('Status')
plt.ylabel('Frequency')
ax.set_xticklabels(labels, ha = 'center', rotation = 'horizontal')
plt.show()

smote = SMOTE(sampling_strategy='minority', k_neighbors=1)
X_sm, y_sm = smote.fit_sample(X, Y)

plot_2d_space(X_sm, y_sm, 'Balanced by SMOTE')

X = X_sm
Y = y_sm
labels1 = ['Alive', 'Expired']
fig = plt.figure()
ax = fig.add_subplot(111)
pd.value_counts(Y).plot.bar()
plt.title('Mortality Histogram')
plt.xlabel('Status')
plt.ylabel('Frequency')
ax.set_xticklabels(labels1, ha = 'center', rotation = 'horizontal')
plt.show()

# split data into train and test sets
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

# fit model no training data
model = XGBClassifier(learning_rate=0.2, max_depth=64, n_estimators=100)

model.fit(X_train, y_train)

print(model)
print(model.feature_importances_)
# plot
labels2 = list(ML_analysis.columns.values[0:11])

# plot feature importance
mapper = {'f{0}'.format(i): v for i, v in enumerate(labels2)}
mapped = {mapper[k]: v for k, v in model.get_booster().get_score().items()}
plot_importance(mapped, height=0.5, grid=False, xlim=(0, max(mapped.values())*1.15), title="Feature Importance", color='red')
plt.show()

plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.xticks(range(len(model.feature_importances_)), labels2, rotation ='vertical')
plt.title('Feature Importance')
plt.show()

# make predictions for test data
y_pred0 = model.predict(X_test)
y_pred = np.where(y_pred0 == 'Alive', 0, 1)
predictions = [round(value) for value in y_pred]
y_test = np.where(y_test == 'Alive', 0, 1)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
confusion = confusion_matrix(y_test, predictions)
print(confusion)
kappa = cohen_kappa_score(y_test, predictions)
print('''Cohen's kappa: %f''' % kappa)

labels = ['Alive', 'Expired']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion, cmap=plt.cm.GnBu)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels, rotation ='vertical', va = 'center')
plt.xlabel('Predicted')
plt.ylabel('Expected')
for (i, j), z in np.ndenumerate(confusion):
    ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')
plt.tick_params(axis='x', bottom=False)
plt.show()

report = classification_report(y_test, predictions, target_names=labels)
print(report)

###############################################################################
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('xgboost: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill: AUC=%.3f' % (ns_auc))
plt.plot(lr_fpr, lr_tpr, marker='.', label='xgboost: AUC=%.3f' % (lr_auc))
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
# show the legend
plt.legend(frameon=False, loc='lower right')
# show the plot
plt.show()
###############################################################################
# precision-recall curve and f1
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

# predict probabilities
lr_probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
yhat = np.where(yhat == 'Alive', 0, 1)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('xgboost: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='xgboost: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend(frameon=False)
# show the plot
plt.show()
###############################################################################
