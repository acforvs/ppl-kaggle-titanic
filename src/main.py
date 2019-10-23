import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

#setting constants:
DEMO_FILE = 'demo.txt'
SEPARATOR = '########################################################################################################################################################################################################'
SEPARATOR = '\n\n'+ SEPARATOR + '\n\n'
f = open(DEMO_FILE, 'a')
#getting data:
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
td = pd.concat([train, test], ignore_index=True, sort=False)
#Family colomn:
td['Family'] = td.Parch + td.SibSp
print(td.head(), SEPARATOR, file=f)
#Whether has family or not colomn:
td['Is_Alone'] = td.Family == 0
print('IS_ALONE:\n', td.head(), SEPARATOR, file=f)
#trying to group tickets by Fare into 4 groups:
td['Fare_Category'] = pd.cut(td['Fare'], bins=[0, 7.90, 14.5, 35.49, 135.66, 153.4], labels=['Low', 'Mid', 'High_Low', 'High_Mid', 'High'])
print('FARE_CATEGORY:\n', td.head(), SEPARATOR, file=f)
#filling missing data:
#the most common Emarked value:
td.Embarked.fillna(td.Embarked.mode()[0], inplace=True)
#NotStated value for the empty fields in Cabin colomn:
td.Cabin = td.Cabin.fillna('NaN')
print('CABIN:\n', td.head(), SEPARATOR, file=f)
#Grouping Age
td['Age_Range'] = pd.cut(td.Age, [0, 16, 32, 48, 64])
#missing Age:
td['Appeal'] = td.Name.apply(lambda name : name.split(',')[1].split('.')[0].strip())
grp = td.groupby(['Sex', 'Pclass'])
grp.Age.apply(lambda x : x.fillna(x.median()))
td.Age.fillna(td.Age.median, inplace=True)
td.Age.fillna(td.Age.median, inplace=True)
#Creating the data for prediction:
td = pd.concat([td, pd.get_dummies(td.Cabin, prefix="Cabin"), pd.get_dummies(td.Age_Range, prefix="Age_Range"), pd.get_dummies(td.Embarked, prefix="Emb", drop_first=True), pd.get_dummies(td.Appeal, prefix="Title", drop_first=True), pd.get_dummies(td.Fare_Category, prefix="Fare", drop_first = True), pd.get_dummies(td.Pclass, prefix="Class", drop_first = True)], axis=1)
td['Sex'] = LabelEncoder().fit_transform(td['Sex'])
td['Is_Alone'] = LabelEncoder().fit_transform(td['Is_Alone'])
print('FINAL DATA:\n', td.head(), SEPARATOR, file=f)
#Dropping data:
td.drop(['Pclass', 'Fare', 'Cabin', 'Fare_Category', 'Name', 'Appeal', 'Ticket', 'Embarked', 'Age_Range', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)

#prediction
X_to_be_predicted = td[td.Survived.isnull()]
X_to_be_predicted = X_to_be_predicted.drop(['Survived'], axis = 1)

train_data = td
train_data = train_data.dropna()
feature_train = train_data['Survived']
label_train  = train_data.drop(['Survived'], axis = 1)
train_data.shape

clf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

x_train, x_test, y_train, y_test = train_test_split(label_train, feature_train, test_size=0.2)
clf.fit(x_train,  np.ravel(y_train))
print("Accuracy: " + repr(round(clf.score(x_test, y_test) * 100, 2)) + "%", SEPARATOR, file=f)
result_rf = cross_val_score(clf, x_train,y_train, cv=10, scoring='accuracy')
y_pred = cross_val_predict(clf, x_train, y_train, cv=10)

result = clf.predict(X_to_be_predicted)
submission = pd.DataFrame({'PassengerId' : X_to_be_predicted.PassengerId, 'Survived' : result})
submission.Survived = submission.Survived.astype(int)
filename = 'to_submit(1).csv'
submission.to_csv(filename, index=False)

f.close()