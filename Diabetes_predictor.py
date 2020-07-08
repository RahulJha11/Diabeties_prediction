import numpy as np
import pandas as pd
import pickle
import seaborn as sb
import seaborn as sns
from pyexpat import model
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Loading Data

df = pd.read_csv('C:\\Users\\RAHUL\\Desktop\\DataSets\\Kaggle_Diabetes_data.csv')

# Renaming DiabetesPedigreeFunction as DPF

df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 value from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_new = df.copy(deep=True)
df_new[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_new[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution

df_new['Glucose'].fillna(df_new['Glucose'].mean(), inplace= True)
df_new['BloodPressure'].fillna(df_new['BloodPressure'].mean(), inplace= True)
df_new['SkinThickness'].fillna(df_new['SkinThickness'].median(), inplace= True)
df_new['Insulin'].fillna(df_new['Insulin'].median(), inplace= True)
df_new['BMI'].fillna(df_new['BMI'].median(), inplace= True)

# Model building
# compare Machine Learning Algorithms Consistently

X = df_new.drop(columns='Outcome')
Y = df_new['Outcome']
# seed = 7
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('RF', RandomForestClassifier()))
# models.append(('SVM', SVC()))
#
# # Evaluate each model in tern
# results = []
# names = []
# scoring = 'accuracy'
# for name, model in models:
#     kflod = model_selection.KFold(n_splits=10, random_state= seed)
#     cv_results = model_selection.cross_val_score(model, X, Y, cv= kflod, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     MSG = "%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
#     print(MSG)

# from the above model we got SVM is the best model with an accuracy of 99.6%

# Creating SVM model
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
classifier = SVC()
classifier.fit(X_train, y_train)

# Create a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))