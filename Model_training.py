import pandas as pd 
import numpy as np 
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error , root_mean_squared_error , r2_score



#Load the dataset
df = pd.read_csv("data/titanic_cleaned_dataset.csv")
print(df)

#Drop the unnecessary columns 
cols_to_drop = df.drop(columns=["Name" , "Ticket" , "Unnamed: 0" , "PassengerId" ,"Sex" ,  "Age" , "Embarked"] , inplace=True)
print("-" * 100)

#Define the independent and dependent feature

X = df[[ "Pclass" , "SibSp" , "Parch" ]]
y = df["Fare"]

#train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size= 0.25)


#Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Now apply the linear regression algorithm
regression = LinearRegression()
regression.fit(X_train , y_train)

#Predictions 
y_pred = regression.predict(X_test)


#Performance matrix
mae = mean_absolute_error(y_test , y_pred)
mse = mean_squared_error(y_test , y_pred)
rmse = root_mean_squared_error(y_test , y_pred)
score = r2_score(y_test , y_pred)

print("MAE : " , mae)
print("MSE : " , mse)
print("RMSE : " , rmse)
print("R2 score : " , score)

#Save the files 
joblib.dump(regression , "models/model.pkl")
joblib.dump(scaler , "models/scaler.pkl")
print("Files saved Successfully.")