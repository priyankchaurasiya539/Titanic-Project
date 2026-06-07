import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

#Load the new dataset
df = pd.read_csv("data/titanic_cleaned_dataset.csv")
print(df)


"""Visualization"""
#Now check the number of males and females 

gender_counts = df["Sex"].value_counts()
plt.bar(gender_counts.index , gender_counts.values,  color=["lightpink" , "lightgreen"], width=0.6 , edgecolor = "black")
plt.xlabel("Gender")
plt.ylabel("Number of persons")
plt.savefig("Graphs(EDA)/Gender ratio(bar).png")
plt.show()

plt.pie(gender_counts.values , labels=gender_counts.index , autopct="%1.2f%%" , colors=["lightpink" , "lightgreen"] )
plt.title("Distribution of sex on the titanic ship")
plt.savefig("Graphs(EDA)/Gender ratio(pie).png")
plt.show()


#Now check the survived rate
survived_counts = df["Survived"].value_counts()
plt.bar(survived_counts.index , survived_counts.values , color=["lightcoral" , "magenta"] , width=0.4 ,  edgecolor = "black")
plt.xlabel("Survived")
plt.ylabel("Number of persons")
plt.xticks([0,1] , ["Not Survived" , "Survived"])
plt.savefig("Graphs(EDA)/Survived Rate(bar).png")
plt.show()

plt.pie(survived_counts.values , labels=survived_counts.index , autopct="%1.2f%%" , colors=["lightcoral" , "magenta"])
plt.title("Distribution of survived rate")
plt.savefig("Graphs(EDA)/Survived Rate(pie).png")
plt.show()

#Distribution of age group
plt.hist(df["Age"] , bins=72 , color = "orange" , edgecolor = "black")
plt.xlabel("Ages")
plt.ylabel("Number of persons")
plt.title("Distribution of ages")
plt.savefig("Graphs(EDA)/Age Distribution.png")
plt.show()

#Embarked(Mainly three places were there C = Cherbourg (France) , Q = Queenstown (Ireland) , S = Southampton (England))
sns.countplot(x= "Embarked" , data=df , palette="pastel" , width=0.5 , legend=False  , edgecolor = "black")
plt.title("Embarked Distribution")
plt.savefig("Graphs(EDA)/Embarked.png")
plt.show()

#Now we have to find number of males and females boarded from particular place

#group by embarked and sex and then count 
df["Embarked"] = df["Embarked"].replace({"C":"Cherbourg", "S":"Southampton", "Q":"Queenstown"})
embarked_gender = df.groupby(["Embarked" , "Sex"]).size().unstack()
embarked_gender.plot(kind="bar" , color= ["aquamarine" , "orchid"] , edgecolor="black")
plt.title("Number of Males & Females from the Embarked Port")
plt.xlabel("Embarked Port")
plt.ylabel("Number of Persons")
plt.xticks(rotation=0)
plt.savefig("Graphs(EDA)/Embarked_Sex.png")
plt.show()

#Plot the graph number of males and females survived 
df["Survived"] = df["Survived"].replace({"0" : "Not Survived" , "1" : "Survived"})
survived_sex = df.groupby(["Survived" , "Sex"]).size().unstack()
survived_sex.plot(kind="bar" , color = ["teal" , "mediumseagreen"] , edgecolor= "black")
plt.title("Distribution of Males & Females Survived")
plt.xlabel("Survived")
plt.ylabel("Number of Persons")
plt.xticks(rotation = 0 )
plt.xticks([0,1] , ["Not Survived" , "Survived"])
plt.savefig("Graphs(EDA)/Survived_Sex.png")
plt.show()