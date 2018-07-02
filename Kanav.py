import numpy as np
import pandas as pd

dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')
X = dataset.iloc[:, 0:66].values
y = dataset.iloc[:, 66].values
X_test = testset.iloc[:, 0:66].values
#Data Cleaning  #COMPLETED TILL Z COLUMN IN EXCEL


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 3:4])
X[:, 3:4] = imputer.transform(X[:, 3:4])
imputer = imputer.fit(X_test[:, 3:4])
X_test[:, 3:4] = imputer.transform(X_test[:, 3:4])
'''
imputer = imputer.fit(X_test[:, 42:43])
X_test[:, 42:43] = imputer.transform(X_test[:, 42:43])
'''
imputer = imputer.fit(X[:, 48:49])
X[:, 48:49] = imputer.transform(X[:, 48:49])
imputer = imputer.fit(X_test[:, 48:49])
X_test[:, 48:49] = imputer.transform(X_test[:, 48:49])

temp = X_test[:,2]
for i in range(1459):
    if temp[i] != 'RH' and temp[i]!='RL' and temp[i]!='RM' and temp[i]!='FV'and temp[i]!='C (all)':
        temp[i] = 'c'
X_test[:,2] = temp   

temp = X_test[:,15]
for i in range(1459):
    if temp[i] != 'VinylSd' and temp[i]!='Wd Sdng' and temp[i]!='HdBoard' and temp[i]!='Plywood'and temp[i]!='MetalSd' and temp[i]!='CemntBd' :
        temp[i] = 'c'
X_test[:,15] = temp   

temp = X_test[:,16]
for i in range(1459):
    if temp[i] != 'VinylSd' and temp[i]!='Wd Sdng' and temp[i]!='HdBoard' and temp[i]!='Plywood'and temp[i]!='MetalSd' and temp[i]!='CemntBd' and temp[i]!='ImStucc':
        temp[i] = 'c'
X_test[:,16] = temp  

temp = X[:,17]
for i in range(1460):
    if temp[i] != 'Brkface' and temp[i]!='Stone':
        temp[i] = 'c'
X[:,17] = temp    

temp = X_test[:,17]
for i in range(1459):
    if temp[i] != 'Brkface' and temp[i]!='Stone':
        temp[i] = 'c'
X_test[:,17] = temp   


temp = X[:,21]
for i in range(1460):
    if temp[i] != 'Gd' and temp[i]!='TA' and temp[i]!='Ex':
        temp[i] = 'c'
X[:,21] = temp   

temp = X_test[:,21]
for i in range(1459):
    if temp[i] != 'Gd' and temp[i]!='TA' and temp[i]!='Ex':
        temp[i] = 'c'
X_test[:,21] = temp   


temp = X[:,22]
for i in range(1460):
    if temp[i] != 'No' and temp[i]!='Gd' and temp[i]!='Av' and temp[i]!='Mn':
        temp[i] = 'c'
X[:,22] = temp   


temp = X_test[:,22]
for i in range(1459):
    if temp[i] != 'No' and temp[i]!='Gd' and temp[i]!='Av' and temp[i]!='Mn':
        temp[i] = 'c'
X_test[:,22] = temp   


temp = X[:,23]
for i in range(1460):
    if temp[i] != 'GLQ' and temp[i]!='ALQ' and temp[i]!='Unf' and temp[i]!='Rec'and temp[i]!='BLQ' and temp[i]!='LwQ':
        temp[i] = 'c'
X[:,23] = temp   

temp = X_test[:,23]
for i in range(1459):
    if temp[i] != 'GLQ' and temp[i]!='ALQ' and temp[i]!='Unf' and temp[i]!='Rec'and temp[i]!='BLQ' and temp[i]!='LwQ':
        temp[i] = 'c'
X_test[:,23] = temp   



temp = X[:,25]
for i in range(1460):
    if temp[i] != 'GLQ' and temp[i]!='ALQ' and temp[i]!='Unf' and temp[i]!='Rec'and temp[i]!='BLQ' and temp[i]!='LwQ':
        temp[i] = 'c'
X[:,25] = temp 

temp = X_test[:,25]
for i in range(1459):
    if temp[i] != 'GLQ' and temp[i]!='ALQ' and temp[i]!='Unf' and temp[i]!='Rec'and temp[i]!='BLQ' and temp[i]!='LwQ':
        temp[i] = 'c'
X_test[:,25] = temp 


temp = X_test[:,42]
for i in range(1459):
    if temp[i] != 'TA' and temp[i]!='Gd' and temp[i]!='Ex' and temp[i]!='Fa':
        temp[i] = 'c'
X_test[:,42] = temp 


temp = X_test[:,44]
for i in range(1459):
    if temp[i] != 'Typ' and temp[i]!='Min2' and temp[i]!='Min1' and temp[i]!='Mod':
        temp[i] = 'c'
X_test[:,44] = temp 


temp = X[:,46]
for i in range(1460):
    if temp[i] != 'TA' and temp[i]!='ALQ' and temp[i]!='Gd' and temp[i]!='Fa':
        temp[i] = 'c'
X[:,46] = temp 


temp = X_test[:,46]
for i in range(1459):
    if temp[i] != 'TA' and temp[i]!='ALQ' and temp[i]!='Gd' and temp[i]!='Fa':
        temp[i] = 'c'
X_test[:,46] = temp 


temp = X[:,47]
for i in range(1460):
    if temp[i] != 'Attchd' and temp[i]!='Detchd' and temp[i]!='BuiltIn' and temp[i]!='CarPort' and temp[i]!= 'Basement' and temp[i]!='2Types':
        temp[i] = 'c'
X[:,47] = temp 

temp = X_test[:,47]
for i in range(1459):
    if temp[i] != 'Attchd' and temp[i]!='Detchd' and temp[i]!='BuiltIn' and temp[i]!='CarPort' and temp[i]!= 'Basement' and temp[i]!='2Types':
        temp[i] = 'c'
X_test[:,47] = temp 


temp = X[:,49]
for i in range(1460):
    if temp[i] != 'RFn' and temp[i]!='Fin' and temp[i]!='Unf':
        temp[i] = 'c'
X[:,49] = temp 

temp = X_test[:,49]
for i in range(1459):
    if temp[i] != 'RFn' and temp[i]!='Fin' and temp[i]!='Unf':
        temp[i] = 'c'
X_test[:,49] = temp 


temp = X[:,52]
for i in range(1460):
    if temp[i] != 'TA' and temp[i]!='Gd' and temp[i]!='Fa':
        temp[i] = 'c'
X[:,52] = temp 

temp = X_test[:,52]
for i in range(1459):
    if temp[i] != 'TA' and temp[i]!='Gd' and temp[i]!='Fa':
        temp[i] = 'c'
X_test[:,52] = temp 


temp = X[:,59]
for i in range(1460):
    if temp[i] != 'MnPrv' and temp[i]!='GdWo' and temp[i]!='GdPrv':
        temp[i] = 'c'
X[:,59] = temp

temp = X_test[:,59]
for i in range(1459):
    if temp[i] != 'MnPrv' and temp[i]!='GdWo' and temp[i]!='GdPrv':
        temp[i] = 'c'
X_test[:,59] = temp


temp = X[:,60]
for i in range(1460):
    if temp[i] != 'Shed' and temp[i]!='GdWo' and temp[i]!='GdPrv':
        temp[i] = 'c'
X[:,60] = temp

temp = X_test[:,60]
for i in range(1459):
    if temp[i] != 'Shed' and temp[i]!='GdWo' and temp[i]!='GdPrv':
        temp[i] = 'c'
X_test[:,60] = temp


temp = X_test[:,64]
for i in range(1459):
    if temp[i] != 'WD' and temp[i]!='COD' and temp[i]!='New' and temp[i]!='ConLD' and temp[i]!='ConLw':
        temp[i] = 'c'
X_test[:,64] = temp
#LabelEncoding


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X_test[:, 2] = labelencoder_X.fit_transform(X_test[:, 2])

X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X_test[:, 5] = labelencoder_X.fit_transform(X_test[:, 5])

X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
X_test[:, 6] = labelencoder_X.fit_transform(X_test[:, 6])

X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
X_test[:, 7] = labelencoder_X.fit_transform(X_test[:, 7])

X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
X_test[:, 8] = labelencoder_X.fit_transform(X_test[:, 8])

X[:, 9] = labelencoder_X.fit_transform(X[:, 9])
X_test[:, 9] = labelencoder_X.fit_transform(X_test[:, 9])

X[:, 10] = labelencoder_X.fit_transform(X[:, 10])
X_test[:, 10] = labelencoder_X.fit_transform(X_test[:, 10])

X[:, 15] = labelencoder_X.fit_transform(X[:, 15])
X_test[:, 15] = labelencoder_X.fit_transform(X_test[:, 15])

X[:, 16] = labelencoder_X.fit_transform(X[:, 16])
X_test[:, 16] = labelencoder_X.fit_transform(X_test[:, 16])

X[:, 17] = labelencoder_X.fit_transform(X[:, 17])
X_test[:, 17] = labelencoder_X.fit_transform(X_test[:, 17])

X[:, 19] = labelencoder_X.fit_transform(X[:, 19])
X_test[:, 19] = labelencoder_X.fit_transform(X_test[:, 19])

X[:, 20] = labelencoder_X.fit_transform(X[:, 20])
X_test[:, 20] = labelencoder_X.fit_transform(X_test[:, 20])

X[:, 21] = labelencoder_X.fit_transform(X[:, 21])
X_test[:, 21] = labelencoder_X.fit_transform(X_test[:, 21])

X[:, 22] = labelencoder_X.fit_transform(X[:, 22])
X_test[:, 22] = labelencoder_X.fit_transform(X_test[:, 22])

X[:, 23] = labelencoder_X.fit_transform(X[:, 23])
X_test[:, 23] = labelencoder_X.fit_transform(X_test[:, 23])

X[:, 25] = labelencoder_X.fit_transform(X[:, 25])
X_test[:, 25] = labelencoder_X.fit_transform(X_test[:, 25])

X[:, 29] = labelencoder_X.fit_transform(X[:, 29])
X_test[:, 29] = labelencoder_X.fit_transform(X_test[:, 29])

X[:, 30] = labelencoder_X.fit_transform(X[:, 30])
X_test[:, 30] = labelencoder_X.fit_transform(X_test[:, 30])

X[:, 31] = labelencoder_X.fit_transform(X[:, 31])
X_test[:, 31] = labelencoder_X.fit_transform(X_test[:, 31])

X[:, 42] = labelencoder_X.fit_transform(X[:, 42])
X_test[:, 42] = labelencoder_X.fit_transform(X_test[:, 42])

X[:, 44] = labelencoder_X.fit_transform(X[:, 44])
X_test[:, 44] = labelencoder_X.fit_transform(X_test[:, 44])

X[:, 46] = labelencoder_X.fit_transform(X[:, 46])
X_test[:, 46] = labelencoder_X.fit_transform(X_test[:, 46])

X[:, 47] = labelencoder_X.fit_transform(X[:, 47])
X_test[:, 47] = labelencoder_X.fit_transform(X_test[:, 47])

X[:, 49] = labelencoder_X.fit_transform(X[:, 49])
X_test[:, 49] = labelencoder_X.fit_transform(X_test[:, 49])

X[:, 52] = labelencoder_X.fit_transform(X[:, 52])
X_test[:, 52] = labelencoder_X.fit_transform(X_test[:, 52])

X[:, 53] = labelencoder_X.fit_transform(X[:, 53])
X_test[:, 53] = labelencoder_X.fit_transform(X_test[:, 53])

X[:, 59] = labelencoder_X.fit_transform(X[:, 59])
X_test[:, 59] = labelencoder_X.fit_transform(X_test[:, 59])

X[:, 60] = labelencoder_X.fit_transform(X[:, 60])
X_test[:, 60] = labelencoder_X.fit_transform(X_test[:, 60])

X[:, 64] = labelencoder_X.fit_transform(X[:, 64])
X_test[:, 64] = labelencoder_X.fit_transform(X_test[:, 64])

X[:, 65] = labelencoder_X.fit_transform(X[:, 65])
X_test[:, 65] = labelencoder_X.fit_transform(X_test[:, 65])

#FEATURE SCALING 

imputer = imputer.fit(X[:, 3:4])
X[:, 3:4] = imputer.transform(X[:, 3:4])

imputer = imputer.fit(X_test[:, 3:4])
X_test[:, 3:4] = imputer.transform(X_test[:, 3:4])


imputer = imputer.fit(X[:, 18:19])
X[:, 18:19] = imputer.transform(X[:, 18:19])

imputer = imputer.fit(X_test[:, 18:19])
X_test[:, 18:19] = imputer.transform(X_test[:, 18:19])


imputer = imputer.fit(X[:, 48:49])
X[:, 48:49] = imputer.transform(X[:,48:49])

imputer = imputer.fit(X_test[:, 48:49])
X_test[:, 48:49] = imputer.transform(X_test[:,48:49])

X = X[:,1:66]
X_test = X_test[:, 1:66]

checker = []
row = 0
import numbers
for i in X_test:
    counter= 0
    for j in i:
        if isinstance(j, numbers.Rational) == False:
            if counter not in checker:
                imputer = imputer.fit(X_test[:, counter:(counter+1)])
                X_test[:, counter:(counter+1)] = imputer.transform(X_test[:, counter:(counter+1)])
                checker.append(counter)
        counter = counter +1
    row = row +1

#Prediction of Results
'''
from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X_test)
'''
'''
FAIL #FeatureSelection
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1460,1)).astype(object), values = X,axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
'''
from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile = 50)
select.fit(X,y)
X_new = select.transform(X)
mask = select.get_support()
#   print(mask)
X_train_opt = X[:, [1,3,10,12,13,16,17,18,19,20,21,23,26,27,28,30,31,32,34,37,38,41,42,44,45,46,47,48,49,50,54,60]]
X_test_opt = X_test[:, [1,3,10,12,13,16,17,18,19,20,21,23,26,27,28,30,31,32,34,37,38,41,42,44,45,46,47,48,49,50,54,60]]

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100,random_state = 0)
regressor.fit(X_train_opt,y)
y_pred = regressor.predict(X_test_opt)



df = pd.DataFrame(y_pred, columns=["SalePrice"])
df.to_csv('RandomForest.csv', index=False)


