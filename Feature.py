import pandas as pd 


df = pd.read_csv("data/titanic_dataset.csv")
print(df.head(10))
print("-" * 100)
print("Shape : " , df.shape)
print("-" * 100)
#Now checking the missing values
print(df.isnull().sum())
print("-" * 100)

#Checking the different types of embarked in the dataset
print(df["Embarked"].unique())
print("-" * 100)
df["Embarked"] = df["Embarked"].fillna("S")
print("-" * 100)

#Filling the age column with the most frequent value
#First check the most frequent values
print(df["Age"].unique())
print("-" * 100)
df["Age"] = df["Age"].fillna(df["Age"].mode()[0])

# print(df.isnull().sum())

#Converting some columns(age , fare) into integer type
df[["Age"]] = df[["Age"]].astype(int)
print(df.info())
print("-" * 100)
print(df.columns)
print(df.isnull().sum())

#Remove the cabin ( because it is unnecessary for the ml model)
df = df.drop(columns= ["Cabin"])
print(df.head(10))

#Save the cleaned csv file 
df.to_csv("titanic_cleaned_dataset.csv")
print("-" * 100)
print("File saved successfully.")
