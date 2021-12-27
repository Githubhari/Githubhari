# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:25:12 2021

@author: komma
"""

from pandas import *
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from numpy import inf,sqrt

#extracting data from the folder
housing_data=read_csv("F:/programming book/handson-ml2-master/datasets/housing/housing1.csv")

#creating extra feature for spliting pourpouse
housing_data["income_cat"]=cut(housing_data["median_income"], bins=[0,1.5,
                                                                    3.0,4.5,6.0,inf]
                               ,labels=[1,2,3,4,5])

#spliting housing_data into.Train_set,test_set
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_set,test_set in split.split(housing_data, housing_data["ocean_proximity"]):
    strat_train_set1=housing_data.iloc[train_set]
    strat_test_set1=housing_data.iloc[test_set]

for train_set,test_set in split.split(housing_data, housing_data["income_cat"]):
    strat_train_set2=housing_data.iloc[train_set]
    strat_test_set2=housing_data.iloc[test_set]

#droping the income_cat from the both strat_train_set,strat_test_set
for set_ in (strat_train_set1,strat_test_set1,strat_train_set2,strat_test_set2):
    set_.drop("income_cat",axis=1,inplace=True)
    
housing_label1=strat_train_set1["median_house_value"].copy()
housing1=strat_train_set1.drop("median_house_value",axis=1)
housing_label2=strat_train_set2["median_house_value"].copy()
housing2=strat_train_set2.drop("median_house_value",axis=1)
housing_num=housing1.drop("ocean_proximity",axis=1)

#creating pipeline for the preprocessing porpous
num_pip=Pipeline([
    ("imputer",SimpleImputer(strategy="median"))
    ,("std_scalar",StandardScaler())])
num_attrib=list(housing_num)
cat_attrib=["ocean_proximity"]
full_pip=ColumnTransformer([
    ("num",num_pip,num_attrib)
    ,("cat",OneHotEncoder(),cat_attrib)])

#using pipeline and processed the housing data's
housing_prepared1=full_pip.fit_transform(housing1)
housing_prepared2=full_pip.fit_transform(housing2)

#creating  LinearRegression models
line_model1=LinearRegression()
line_model2=LinearRegression()
line_model1.fit(housing_prepared1,housing_label1)
line_model2.fit(housing_prepared2,housing_label2)
pred1=line_model1.predict(housing_prepared1)
pred2=line_model2.predict(housing_prepared2)
err1=mean_squared_error(housing_label1,pred1)
err2=mean_squared_error(housing_label2,pred2)
print(sqrt(err1),sqrt(err2))
#cross-validation for above two models
linescore1=cross_val_score(line_model1,housing_prepared1,housing_label1,scoring="neg_mean_squared_error",cv=5)
linescore2=cross_val_score(line_model2,housing_prepared2,housing_label2,scoring="neg_mean_squared_error",cv=5)
print(sqrt(-linescore1),sqrt(-linescore2))
#from the above cross-validation i can say,that this data is more complix then
#this model(underfitting happens)

#creating DecisionTreeRegressor models
tree1=DecisionTreeRegressor()
tree2=DecisionTreeRegressor()
tree1.fit(housing_prepared1,housing_label1)
tree2.fit(housing_prepared2,housing_label2)
treepred1=tree1.predict(housing_prepared1)
treepred2=tree2.predict(housing_prepared2)
treeerr1=mean_squared_error(housing_label1,treepred1)
treeerr2=mean_squared_error(housing_label2,treepred2)
print(sqrt(treeerr1),sqrt(treeerr2))
#cross-validation for above two models
treescore1=cross_val_score(tree1,housing_prepared1,housing_label1,scoring="neg_mean_squared_error",cv=5)
treescore2=cross_val_score(tree2,housing_prepared2,housing_label2,scoring="neg_mean_squared_error",cv=5)
print(sqrt(-treescore1),sqrt(-treescore2))
#from the above cross-validation i can say that this model is more complix to
#this data(overfitting happens more)

#creating RandomForestRegressor models
forest1=RandomForestRegressor()
forest2=RandomForestRegressor()
forest1.fit(housing_prepared1,housing_label1)
forest2.fit(housing_prepared2,housing_label2)
forestpred1=forest1.predict(housing_prepared1)
forestpred2=forest2.predict(housing_prepared2)
foresterr1=mean_squared_error(housing_label1,forestpred1)
foresterr2=mean_squared_error(housing_label2,forestpred2)
print(sqrt(foresterr1),sqrt(foresterr2))
#cross-validation for above two models
forestscore1=cross_val_score(forest1,housing_prepared1,housing_label1,scoring="neg_mean_squared_error",cv=5)
forestscore2=cross_val_score(forest2,housing_prepared2,housing_label2,scoring="neg_mean_squared_error",cv=5)
print(sqrt(-forestscore1),sqrt(-forestscore2))
#from the above cross-validation i can say,that this model is not more complix
#to this data(overfitting happens less)

#use GridSearchCV for the RandomForest models to regularize there hyperparams