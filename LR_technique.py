import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# calculate heuristic class weighting
# from sklearn.utils.class_weight import compute_class_weight

ML_prim = pd.read_csv('ML_ready.csv')
ML_analysis = copy.deepcopy(ML_prim)
ML_analysis = ML_analysis.drop(['Unnamed: 0', 'uniquepid'], axis=1)

X = ML_analysis.values[:,0:11]
Y = ML_analysis.values[:,11]

smote = SMOTE(sampling_strategy='minority', k_neighbors=1)
X_sm, y_sm = smote.fit_sample(X, Y)

X = X_sm
Y = y_sm
Y = np.where(Y == 'Alive', 0, 1)
# weighting = compute_class_weight('balanced', [0,1], Y)
# print(weighting)
# split data into train and test sets
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

# define model
# weights = {0:1.0, 1:10}#round(len(y_test[y_test == 0])/len(y_test[y_test == 1]), 2)}
model = LogisticRegression(solver='newton-cg', penalty='l2', class_weight='balanced', random_state=42) # liblinear: , fit_intercept=True, intercept_scaling=6, verbose=1)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)#StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# evaluate model
scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f (%.3f)' % (scores.mean(), scores.std()))

model.fit(X_train, y_train)

# get importance
labels2 = list(ML_analysis.columns.values[0:11])
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.xticks(range(len(importance)), labels2, rotation ='vertical')
plt.title('Feature Importance')
plt.show()

# make predictions for test data
predictions = model.predict(X_test)

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
print('SVM: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill: AUC=%.3f' % (ns_auc))
plt.plot(lr_fpr, lr_tpr, marker='.', label='LR: AUC=%.3f' % (lr_auc))
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
#yhat = np.where(yhat == 'Alive', 0, 1)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('SVM: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='LR: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend(frameon=False)
# show the plot
plt.show()