import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


# setting constants:
DEMO_FILE = 'demo.txt'
SEPARATOR = '########################################################################################################################################################################################################'
SEPARATOR = '\n\n' + SEPARATOR + '\n\n'


def main():
    f = open(DEMO_FILE, 'a')
    # getting data:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    td = pd.concat([train, test],
                   ignore_index=True,
                   sort=False)
    # Family column:
    td['Family'] = td.Parch + td.SibSp
    print('FAMILY:\n', td.head(), SEPARATOR, file=f)
    # Whether has family or not column:
    td['Is_Alone'] = td.Family == 0
    print('IS_ALONE:\n', td.head(), SEPARATOR, file=f)
    # trying to group tickets by Fare into 4 groups:
    td['Fare_Category'] = pd.cut(td['Fare'], 
                                bins=[0, 7.90, 14.5, 35.49, 135.66, 153.4], 
                                labels=['Low', 'Mid', 'High_Low', 'High_Mid', 'High'])
    print('FARE_CATEGORY:\n', td.head(), SEPARATOR, file=f)
    # filling missing data:
    # 1. the most common Emarked value:
    td.Embarked.fillna(td.Embarked.mode()[0], inplace=True)
    # 2. NotStated value for the empty fields in Cabin column:
    td.Cabin = td.Cabin.fillna('NaN')
    print('CABIN:\n', td.head(), SEPARATOR, file=f)
    # 3. missing Age:
    td['Age_Range'] = pd.cut(td['Age'], 
                            [0, 5, 12, 18, 35, 60, 100], 
                            labels=['Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])
    print('AGE_RANGE:\n', td.head(), SEPARATOR, file=f)
    # Mr. / Mrs. / Miss and so on...:
    td['Appeal'] = td.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
    grp = td.groupby(['Sex', 'Pclass', 'Is_Alone'])
    grp.Age.apply(lambda x: x.fillna(x.median()))
    td.Age.fillna(td.Age.median, inplace=True)
    print('AGE_MEDIAN:\n', td.head(), SEPARATOR, file=f)

    # Creating the data for prediction:
    columns_list = [
        'Age_Range',
        'Appeal',
        'Fare_Category',
        'Pclass',
        'Cabin',
        'Ticket',
        'Embarked']
    for column in columns_list:
        dummies = pd.get_dummies(td[column], prefix=column)
        td = pd.concat([td, dummies], axis=1)

    td['Sex'] = LabelEncoder().fit_transform(td['Sex'])
    td['Is_Alone'] = LabelEncoder().fit_transform(td['Is_Alone'])

    # Dropping data:
    dropped_columns = [
        'Pclass',
        'Fare',
        'Cabin',
        'Fare_Category',
        'Name',
        'Appeal',
        'Ticket',
        'Embarked',
        'Age_Range',
        'SibSp',
        'Parch',
        'Age']
    td.drop(dropped_columns,
            axis=1,
            inplace=True)
    print('FINAL DATA:\n', td.head(), SEPARATOR, file=f)

    # prediction
    X_to_be_predicted = td[td.Survived.isnull()]
    X_to_be_predicted = X_to_be_predicted.drop(['Survived'], axis=1)

    train_data = td
    train_data = train_data.dropna()
    feature_train = train_data['Survived']
    label_train = train_data.drop(['Survived'], axis=1)
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
    clf.fit(x_train, np.ravel(y_train))
    ACCURACY = 'Accuracy: ' + repr(round(clf.score(x_test, y_test) * 100, 2)) + '%'
    print(ACCURACY, SEPARATOR, file=f)

    result = clf.predict(X_to_be_predicted)
    submission = pd.DataFrame({'PassengerId': X_to_be_predicted.PassengerId, 'Survived': result})
    submission.Survived = submission.Survived.astype(int)
    filename_to_submit = 'to_submit(1).csv'
    submission.to_csv(filename_to_submit, index=False)

    f.close()


if __name__ == '__main__':
    main()
