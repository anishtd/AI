# -*- coding: utf-8 -*-
"""

@author: anish
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('train.csv')

df.head()

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

df.shape
df.info()

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.drop(['Alley'],axis=1,inplace=True)

df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

df.shape
df.drop(['Id'],axis=1,inplace=True)
df.isnull().sum()

df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')

df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df.dropna(inplace=True)

df.shape
df.head()
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

len(columns)
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
main_df=df.copy()
test_df=pd.read_csv('formulatedtest.csv')

test_df
test_df.head()
final_df=pd.concat([df,test_df],axis=0)

final_df['SalePrice']
final_df.shape
final_df=category_onehot_multcols(columns)

final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]

final_df.shape

df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]

df_Train.head()

df_Test.head()

df_Train.shape

df_Test.drop(['SalePrice'],axis=1,inplace=True)

X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']



import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ReLU,LeakyReLU,PReLU,ELU
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units=50, kernel_initializer='he_uniform', activation='relu', input_dim=174))

classifier.add(Dense(units=25, kernel_initializer='he_uniform', activation='relu'))

classifier.add(Dense(units=50, kernel_initializer='he_uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='he_uniform'))

classifier.compile(loss=root_mean_square_error, optimizer='Adamax')

model_history = classifier.fit(X_train.values, y_train.values, validation_split=0.20,batch_size=10,epochs=1000)

from keras import backend as K
def root_mean_square_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

ann_pred=classifier.predict(df_Test.values)

from sklearn.metrics import mean_squared_error as MSE
rmse_test = MSE(y_test, ann_pred)**(1/2)

history = model_history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['loss'][-1], loss=history['val_loss'][-1]))

pred=pd.DataFrame(ann_pred)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submissions.csv',index=False)
----------------------------------------------------------------------------

from sklearn.feature_selection import VarianceThreshold
var_thred = VarianceThreshold(threshold=0.02)
var_thred.fit(X_train)

sum(var_thred.get_support())
const_columns = [column for column in X_train.columns
                 if column not in X_train.columns[var_thred.get_support()]]
print(const_columns)
len(const_columns)

x_t = x_t.drop(const_columns, axis=1)
x_t = x_t.drop(corr_features, axis=1)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,18))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features = correlation(x_t, 0.7)
len(set(corr_features))
set(corr_features)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, input_dim=x_t.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.2))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.2))
            
        model.add(Dense(units=1, kernel_initializer="he_uniform"))
        model.compile(optimizer='Adamax', loss=root_mean_square_error)
        
        return model
    
    
#layers = [(20,), (40,20), (45,30,15), (60,50,30,20,10), (100,80,50,20,10),(200,160,120,80,60,40,30,20,10)]
layers = [(40,20), (45,30,15), (60,50,30,20,10),(200,160,120,80,60,40,30,20,10)]
layers = []
layers = [(200,160,120,80,60,40,30,20,10)]
#activations = ['sigmoid','relu']
activations = ['relu']
#param_grid = dict(layers=layers, activation= activations, batch_size=[32,64,128,256], epochs=[1000])
param_grid = dict(layers=layers, activation= activations, epochs=[1000])
model = KerasRegressor(build_fn=create_model, verbose=1)

grid=GridSearchCV(estimator=model, param_grid=param_grid,cv=10)

grid_result = grid.fit(x_t, y_train)


df_t = df_Test
df_t = df_t.drop(const_columns, axis=1)
df_t = df_t.drop(corr_features, axis=1)

 
#model_history = classifier.fit(X_train.values, y_train.values, validation_split=0.20,batch_size=10,epochs=1000)

from keras import backend as K
def root_mean_square_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

ann_pred_new=grid.predict(df_t.values)

y_test = pd.read_csv('sample_submission.csv')
y_test = y_test['SalePrice']


from sklearn.metrics import mean_squared_error as MSE
rmse_test = MSE(y_test, ann_pred_new)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, ann_pred_new), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, ann_pred_new), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, ann_pred_new), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, ann_pred_new), 2)) 
print("R2 score =", round(sm.r2_score(y_test, ann_pred_new), 2))

#history = model_history.history
#print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['loss'][-1], loss=history['val_loss'][-1]))

pred=pd.DataFrame(ann_pred_new)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission_24NN.csv',index=False)
