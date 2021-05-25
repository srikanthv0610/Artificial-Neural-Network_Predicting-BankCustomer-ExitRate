###Artificial Neural Network (ANN) for classification###
# Objective: Predicting a Bank's Customers Exit (Churn) Rate using an ANN model

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


### Data Preprocessing

# Loading the data
bank_dataset = pd.read_csv("Learning_Dataset/Bank_data.csv")
print(bank_dataset.head(5))
print(bank_dataset.shape)

# Taking  all rows and all columns in the data except the last column as X (feature matrix) and the row numbers,
# customer id and surname rows since they are not necessary.
X = bank_dataset.iloc[:,3:-1].values
print("Independent variables are:", X)
# Taking all rows of the last column as Y(dependent variable): Indicating customers who exited the bank.
y = bank_dataset.iloc[:, -1].values
print("Dependent variable is:", y)

#Encoding categorical dataset (Changing string data to numbers)
#Column 1 contains the country names and we convert them to numbers
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#Column 2 contains the genders and we convert them to 0 or 1
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
print(X)

#Converting all the values in X dataset to float
ct = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = np.array(ct.fit_transform(X))
X = X.astype('float64')
print(X)

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#printing the dimensions of each of those snapshots to see amount of rows and columns i each of them
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#Feature Scaling (Getting all the DataSet values to lie between similar scaled values)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Building the model

# Initializing the ANN by calling the Sequential class fromm keras of Tensorflow
ann = tf.keras.models.Sequential()

# Adding INPUT layer to the Sequential ANN by calling Dense class
# Number of Units = 6, Activation Function = Rectifier, Inputs variables = 12
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu', input_dim = 12))

# Adding SECOND layer to the Sequential AMM by calling Dense class
# Number of Units = 6 and Activation Function = Rectifier
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# Adding OUTPUT layer to the Sequential ANN by calling Dense class
# Number of Units = 1 and Activation Function = Sigmoid
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

### Training the model
# Compiling the ANN
# Type of Optimizer = Adam Optimizer, Loss Function =  crossentropy for binary dependent variable, and Optimization is done w.r.t. accuracy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN model on training set
# batch_size = 50, the default value, number of epochs  = 100
ann.fit(X_train, y_train, batch_size = 50, epochs = 100)

# the goal is to use this ANN model to predict the probability of the customer leaving the bank
# Predicting the churn probability for single observations

#Geography: French
#Credit Score:800
#Gender: Female
#Age: 48 years old
#Tenure: 5 years
#Balance: $100000
#Number of Products: 2
#with Credit Card: 1
#Active member: 1
#Estimated Salary: $68000

print(ann.predict(sc.transform([[1, 0, 0, 800, 0, 48, 5, 100000, 2, 1, 1, 68000]])))
print(ann.predict(sc.transform([[1, 0, 0, 800, 0, 48, 5, 100000, 2, 1, 1, 68000]]))>0.5)
# this customer has 28% chance to leave the bank

#show the vector of predictions and real values
#probabilities
y_pred_prob = ann.predict(X_test)

#probabilities to binary
y_pred = (y_pred_prob > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

#Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix", confusion_matrix)
print("Accuracy Score", accuracy_score(y_test, y_pred))

# >> Confusion Matrix: [[1529   66]
#                   [ 205  200]]

# >> Accuracy: 86.5 %