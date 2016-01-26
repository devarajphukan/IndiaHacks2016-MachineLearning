import pandas as pd
import numpy as np
import xgboost as xgb

train_data_df = pd.read_csv('train.csv',delimiter=',',header = None)
test_data_df = pd.read_csv('test.csv',header = None ,delimiter=",")

train_data_df.columns = ['solved_status','skills','solved_count','attempts','user_type','level','accuracy','solved_count_p','error_count','rating']
test_data_df.columns = ['skills','solved_count','attempts','user_type','level','accuracy','solved_count_p','error_count','rating']

myResults = train_data_df['solved_status'] 
myResults = np.array(myResults)

labels_numeric = pd.Series(train_data_df['solved_status'],dtype = "float")
labels_numeric = labels_numeric.astype(np.float)
#print labels_numeric
train_data_df = train_data_df.drop('solved_status',1)

train_data_df = np.array(train_data_df)

test_data_df = np.array(test_data_df)

xg_train = xgb.DMatrix(train_data_df,label=labels_numeric)
xg_test = xgb.DMatrix(test_data_df)

param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['gamma'] = 1
# param['n_estimators'] = 1000
param['min_child_weight'] = 6
param['max_depth'] = 8
param['subsample'] = 0.85
param['colsample_bytree'] = 0.5
param['max_delta_step'] = 20
#param['lambda'] = 10
num_round = 800

gbm = xgb.train(param,xg_train,num_round)
test_pred = gbm.predict(xg_test,output_margin = True)

f = open('new_results7.csv','w')
f.write("Id,solved_status\n")
a = 0
for i in test_pred :
	j = str(i).strip()
	k = list(j)
	if k[0] == '-':
		j = '0'
	else :
		j = '1'
	f.write(str(a)+","+str(j)+"\n")
	a += 1
		
