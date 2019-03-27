#Breast Cancer Detection Classifier Model

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read in Dataset
breast_cancer_data = pd.read_csv("Breast_cancer_data.csv")

#Visualize the data columns
breast_cancer_data.head()

#Summarize the data
breast_cancer_data.describe()

#Visualize the distributions of the variables
sns.distplot(breast_cancer_data['mean_radius'])

sns.distplot(breast_cancer_data['mean_texture'])
#Distribution plot appears to be normal with a slight skew

sns.distplot(breast_cancer_data['mean_perimeter'])
#Distribtuion has skewness

sns.distplot(breast_cancer_data['mean_area'])
#Distribtuion has skewness

sns.distplot(breast_cancer_data['mean_smoothness'])
#Distribtuion appears normal

#View the pair plots for the data to see realtionships between two variables
#In some variables we can see clear linear realtionship
sns.pairplot(breast_cancer_data[['mean_smoothness','mean_area', 'mean_perimeter', 'mean_texture','mean_radius']],palette = sns.color_palette("GnBu_d"), size=2.5)

#Use correlation heat map matrix to see linear realtionship between variables
corr = breast_cancer_data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

#Split the data into independent and dependent variables
X = breast_cancer_data.iloc[:, : -1].values
y = breast_cancer_data.iloc[:, -1].values

X

#Use PCA to reduce variables because some show linear correlation to each other
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

#Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

X

#Use the PCA, to reduce variables. This prevents overfitting as well
X = pca.fit_transform(X)

#Split dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Apply Random Forest Algorithim
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
random_forest.fit(X_train,y_train)
y_predicted = []
y_predicted = random_forest.predict(X_test)

#View accuracy using confusion matrix
confusion_matrix(y_test,y_predicted)

#Print Accuracy of Random Forest Algorithim
print("Accuracy of Random Forest was:", (38 + 68)/ (38+68+3+5))

#Apply Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_predicted = []
y_predicted = naive_bayes.predict(X_test)

#View accuracy using confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predicted)

#Print Accuracy of Naive Bayes Algorithim
print("Accuracy of Naive Bayes was:", (36+70)/ (36+70+1+7))

#Apply KNN Classification
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train,y_train)
y_predicted = []
y_predicted = KNN.predict(X_test)

#View accuracy using confusion matrix
confusion_matrix(y_test,y_predicted)

#Print Accuracy of KNN Algorithim
print("Accuracy of KNN was:", (37 + 68)/ (37+68+3+6))

#Overall all models performed very well with 92% accuracy
