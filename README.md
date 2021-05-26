# Predicting a Bank Customer Exit Rate using ANN:

* The dataset used for modelling our prediction model can be found [here](https://github.com/srikanthv0610/Artificial-Neural-Network_Predicting-BankCustomer-ExitRate/tree/main/Dataset)

* Applying Data Preprocessing: Data Transformation, Data Normalization and Splitting into Train and Test Set

* Artificial Neural Network Modelling, Selecting the model parameters

* Using the ANN model to Predict

* Evaluating using Accuracy score and Confusion Matrix

# Data Transformation:

* First we transform the gender variable to binary. (Female = 0, Male = 1).
* We then use OneShotEncoder to transform the Geography variable tp a categorical variable.

# Data Normalization:

We feature scale the independent variables using Scikit learn: StandardScale.

# Train_Test Split:

We split the original data into 70% train set and 30% test set 


# Plots
![Heatmap](https://github.com/srikanthv0610/Artificial-Neural-Network_Predicting-BankCustomer-ExitRate/blob/main/Plots/Correlation_analysis.png)

![Correlation](https://github.com/srikanthv0610/Artificial-Neural-Network_Predicting-BankCustomer-ExitRate/blob/main/Plots/Correlation_target.png)

Observation from Correlation:

* Tenure and NumOfProduct variables are the least correlated to the exited variable
* Age and Balance variables have the highest complementary correlation with our target(exited) variable
* IsActiveMember and Gender variables have the highest supplementary correlation with the target variable
* Based on Geography: Resident from Germany is more likely to exit than a resident from France or Spain

# ANN Model Evaluation:

>> Confusion Matrix: [[1518   77] [192   33]]
>> Accuracy_Score: 86.5 %





