import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

"""
3.1 Load the Wine dataset into a Pandas DataFrame (HINT: the file is not comma delimited but it is delimited by a ; you can change the delimiter or have pandas accept it)
3.2 Show the Head and Tail of the Dataframe
* 3.3 Provide information about the dataset
* 3.4 Create a correlation Matrix and an accompanying Heat Map
* 3.5 Using the heat map find three strong corelations and plot them using a scatter plot 
* (BONUS 1: Use Matplotlib and Seaborn) 
* (BONUS 2: Include a best fit line)
"""

# Load the Wine dataset into a Pandas DataFrame
data=pd.read_csv('winequality-red.csv', sep=';')

# Show the Head and Tail of the Dataframe
data.head()
data.tail()

# Provide information about the dataset
data.info()
data.describe()

# Create a correlation Matrix and an accompanying Heat Map
data.corr()
fig, ax = plt.subplots(figsize=(12,12))
im = ax.imshow(data.corr(), interpolation='nearest')
fig.colorbar(im, orientation='vertical')

# Using the heat map find three strong corelations and plot them using a scatter plot 
f,ax = plt.subplots(figsize=(8,6))
sns.heatmap(data.corr(), cmap="GnBu", annot=True, linewidths=0.5, fmt= '.1f',ax=ax)
plt.show()

"""
STRONG CORRELATIONS:
1. fixed acid and density
2. citric acid and fixed acidity
3. alcohol and quality
"""

sns.lmplot(
    data=data,
    x="fixed acidity", y="density"
)

# Build and train the Linear Regression Model
# Split the data by putting the target column in an array and dropping the target column in the DataFrame
y = data.loc[:,"quality"].values
x = data.drop(['quality'], axis = 1)

# Split the data into Testing and Training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 123)

# Setup the Sklearn Linear Regression
logreg = linear_model.LogisticRegression(max_iter=100000)

# Train the model using the training dataset
logreg.fit(x_train,y_train)

# Test the modle using the testing dataset
predicted = logreg.predict(x_test)

# Show the accuracy of the model
print("Test accuracy: {} ".format(logreg.score(x_test, y_test)))

# Create a confusion Matrix to identify TP, FP, FN, TN
cf_matrix = confusion_matrix(y_test,predicted)
cf_matrix

# Create example entries to predict
column_names=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
test_prediction_array = [ [6.7,0.32,0.44,2.4,0.061,24,34,0.99484,3.29,0.9,12],
                          [1.9,0.1,0.4,2.4,0.03,24,35,0.99484,3.29,0.9,12] ]
test_prediciton_df = pd.DataFrame(test_prediction_array, columns=column_names)
test_prediciton_df

# Predict
logreg.predict(test_prediciton_df)
