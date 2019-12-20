#Importing libraries
import pandas as pd #For accessing dataset
import numpy as np
import matplotlib.pyplot as plt #For Graph

dataset = pd.read_csv('city1.csv') #Reading from folder where csv file is
print(dataset.shape) #Tells how many rows and columns are in the dataset.

#Plotting data points on a graph
#Manual checking if we can find relationship between the data.
#dataset.plot(x='Year', y='Population', style='o')
#plt.title('Pakistan Population')
#plt.xlabel('Year', 'City')
#plt.ylabel('Population')
#plt.plot(dataset.Year,dataset.City, dataset.Population, color='red', marker='+')
#plt.show()#The grap shows, there is a linear relation between Year and Population.

#Graph with seaborn
import seaborn as sns
sns.regplot(x="Year", y="Population",data=dataset);


#Preparing Data
X = dataset.iloc[:, :2] #Year and City
y = dataset.iloc[:, -1] #Population


#Splitting the dataset, 10% for test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


