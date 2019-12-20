# --------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

data = pd.read_csv(path)

# Explore the data 

print(data.sex.value_counts())
print(data[data['sex']=='Female'].age.mean())
print(round(((data[data['native-country']=='Germany'].shape[0])/(data.shape[0] )*100),4))
# mean and standard deviation of their age
print(data[data.salary == '>50K'].age.mean(), data[data.salary == '>50K'].age.std())
print(data[data.salary == '<=50K'].age.mean(), data[data.salary == '<=50K'].age.std())
# Display the statistics of age for each gender of all the races (race feature).
a = data[data['race']=='Amer-Indian-Eskimo']
a[a['sex']=='Male'].age.max()

data.replace(to_replace =[">50K", "<=50K"],  
                            value = [1 , 0 ]) 
# encoding the categorical features.
#Encoding the categorical features.
x = data.select_dtypes(exclude='int64')
cat_cols = list(x.columns)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
#make_new = enc.fit(df_cats.make)
#make_new
data[cat_cols]=data[cat_cols].apply(LabelEncoder().fit_transform)

# Split the data and apply decision tree classifier
#Spliting features and target variable into X and y respectively.
X = data.iloc[:,:-1]
y = data['salary']
#Split the data X and y into Xtrain,Xtest,ytrain and ytest in the ratio 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Further split the training data into train and validation in 80:20 ratio
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

#Decision Tree Classifier model and calculate the accuracy on validation data as well as on test data.
dec_clf = DecisionTreeClassifier(random_state=0)
dec_clf.fit(X_train,y_train)

dec_score_train =  dec_clf.score(X_train,y_train)
print(dec_score_train)

dec_score_test =  dec_clf.score(X_test,y_test)
print(dec_score_test)

# Perform the boosting task
#erform ensembling using the models Decision Tree Classifier and Logistic Regression, using a VotingClassifier 
#keeping the parameter voting as soft and the calculate the accuracy.
#Different models initialised
log_clf_1 = LogisticRegression(random_state=0)
log_clf_2 = LogisticRegression(random_state=42)
decision_clf1 = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
decision_clf2 = DecisionTreeClassifier(criterion = 'entropy', random_state=42)


#Creation of list of models
Model_List=[('Logistic Regression 1', log_clf_1),
            ('Logistic Regression 2', log_clf_2),
            ('Decision Tree 1', decision_clf1),
            ('Decision Tree 2', decision_clf2)]
voting_clf_soft = VotingClassifier(estimators = Model_List,
                                   voting = 'soft')

#Fitting the data
voting_clf_soft.fit(X_train, y_train)

voting_clf_soft_train = voting_clf_soft.score(X_train,y_train)
print(voting_clf_soft_train)

voting_clf_soft_test = voting_clf_soft.score(X_test,y_test)
print(voting_clf_soft_test)

#With Effect of adding More Tress
decision_clf3 = DecisionTreeClassifier(criterion = 'entropy',random_state=35)
decision_clf4 = DecisionTreeClassifier(criterion = 'entropy', random_state=48)

Model_List=[('Logistic Regression 1', log_clf_1),
            ('Logistic Regression 2', log_clf_2),
            ('Decision Tree 1', decision_clf1),
            ('Decision Tree 2', decision_clf2),
            ('Decision Tree 3', decision_clf3),
            ('Decision Tree 4', decision_clf4)]

voting_clf_soft_train = voting_clf_soft.score(X_train,y_train)
voting_clf_soft_test = voting_clf_soft.score(X_test,y_test)
print(voting_clf_soft_test)

# Checking for Best  best gradient boosting classifier 
from sklearn.ensemble import GradientBoostingClassifier

gbrt1 = GradientBoostingClassifier(max_depth=6, n_estimators=50, learning_rate=0.75, random_state=0)
gbrt1.fit(X_train,y_train)
gbrt1_test_score = gbrt1.score(X_test,y_test)
gbrt1_val_score = gbrt1.score(X_val,y_val)
print(gbrt1_test_score,gbrt1_val_score)

gbrt2 = GradientBoostingClassifier(max_depth=6, n_estimators=100, learning_rate=0.75, random_state=0)
gbrt2.fit(X_train,y_train)
gbrt2_test_score = gbrt2.score(X_test,y_test)
gbrt2_val_score = gbrt2.score(X_val,y_val)
print(gbrt2_test_score,gbrt2_val_score)

# With diffenet Learning rates
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 0.85, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=learning_rate, max_features=10, max_depth=6, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))

#Calculate the classification error for model on the training data, Testing Data and Validation Data
#Best learning_rate=0.1, as result of previous task
from sklearn.metrics import mean_squared_error
n_est = [10, 50, 100]
train_err =[]
test_err =[]
val_err =[]

for n_estimators in n_est:
    gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.1, max_features=10, max_depth=6, random_state=0)
    gb_clf.fit(X_train, y_train)
    pred_train = gb_clf.predict(X_train)
    train_err.append(mean_squared_error(pred_train, y_train))
    pred_test = gb_clf.predict(X_test)
    test_err.append(mean_squared_error(pred_test, y_test))
    pred_val = gb_clf.predict(X_val)
    val_err.append(mean_squared_error(pred_val, y_val))
    
# Plottng training , testing and Validation errorr Vs number of trees
plt.plot(n_est, train_err, color='blue')
plt.plot(n_est, test_err, color='red')
plt.plot(n_est, val_err,  color='green')
plt.legend(['Train','Test','Validation'])
plt.xlabel('Number of Trees')
plt.ylabel('Errors')
plt.show()


#  plot a bar plot of the model's top 10 features with it's feature importance score


#  Plot the training and testing error vs. number of trees



