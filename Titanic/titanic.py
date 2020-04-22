import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seab
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from sklearn import linear_model

import warnings
warnings.filterwarnings('ignore')

def getcalling(name):
	calling = name.split(",")[1].split(".")[0].strip()
	if calling == 'Capt' or calling == 'Col' or calling == 'Major' or calling == 'Dr' or calling == 'Rev':
		return 'A'
	elif calling == 'Don' or calling == 'Sir' or calling == 'the Countess' or calling == 'Dona' or calling == 'Lady':
		return 'B'
	elif calling == 'Mme' or calling == 'Ms' or calling == 'Mrs':
		return 'C'
	elif calling == 'Mlle' or calling == 'Miss':
		return 'D'
	elif calling == 'Mr':
		return 'E'
	elif calling == 'Master'or calling == 'Jonkheer':
		return 'F'
	return 0

def dealTicket(t):
	t = str(t)
	if t[0].isdigit():
		return 0
	else:
		return t[0]

train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")
passenger_id = test_data['PassengerId']
total_data = pd.concat([train_data, test_data], ignore_index = True)

# deal with Age
total_data['calling'] = total_data['Name'].apply(getcalling)

age_data = total_data[['Age','Pclass','Sex','calling','SibSp','Parch']]
age_df = pd.get_dummies(age_data)

known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values

y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=17, n_estimators=1000, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1:])
total_data.loc[(total_data.Age.isnull()), 'Age'] = predictedAges

total_data = total_data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','calling']]
total_data.loc[total_data.Cabin.isnull(), 'Cabin'] = 'U'
total_data.loc[total_data['Cabin'] == 'T', 'Cabin'] = 'U'
total_data['Cabin'] = total_data['Cabin'].str[0]

train = total_data[total_data.Survived.notnull()]
test = total_data[total_data.Survived.isnull()]

train.loc[train.Embarked.isnull(), 'Embarked'] = 'C'
test.loc[test.Fare.isnull(), 'Fare'] = total_data[(total_data['Pclass'] == 3) & (total_data['Embarked'] == 'S')].Fare.median()

train_dummy = pd.get_dummies(train, columns = ['Pclass', 'Sex', 'Cabin','Embarked','calling']).values
test_dummy = pd.get_dummies(test, columns = ['Pclass', 'Sex', 'Cabin','Embarked','calling']).values

ss = preprocessing.StandardScaler()

train_dummy = ss.fit_transform(train_dummy)
test_dummy = ss.fit_transform(test_dummy)

train_input = train_dummy[:, 1:]
test_input = test_dummy[:, 1:]
train_target = train_dummy[:, 0]

lr = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-5)
lr.fit(train_input, train_target.astype(int))

predicts = lr.predict(test_input)
ans = pd.DataFrame({'PassengerId': passenger_id, 'Survived': predicts.astype(np.int32)})
ans.to_csv('./answer.csv', index=False)
