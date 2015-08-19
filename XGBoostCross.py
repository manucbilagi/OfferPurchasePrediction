xgboost_python_path = "/home/xgboost-master/python/"
vowpalwabbit_path = "~/vowpal_wabbit/vowpalwabbit/vw"
sofiaml_path = "~/sofia-ml-read-only/sofia-ml"
data_path = "/home/rachita/Data/" # this needs to be the full path to the data folder

import pprint
import sys, pickle, string, os
sys.path.append(xgboost_python_path)
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from subprocess import call
from sklearn.datasets import dump_svmlight_file
from sklearn.cross_validation import ShuffleSplit
from time import gmtime, strftime
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.cross_validation import KFold



offer_est_weights = [1200988, 1194044, 1197502, 1204576, 1199256, 1199258]


for r in range(1):
	# get data
	train_data = pd.io.parsers.read_csv("./original/train/all_features.csv", sep=" ")

	print("the datatype of test_data")
	print train_data.__class__

	n_samples, n_features=train_data.shape
	print ("n_samples=%d" % n_samples)
	print ("n_features=%d" % n_features)

        	   

	train_label = train_data['label'] #pandas.core.series.Series
	train_label_rep = train_data['repeattrips']
	del train_data['label']
	del train_data['repeattrips']
	
	print "the datatype of train2_label"
	print train_label.__class__

	   	
	# remove features which degrade results
	#del train_data['offer_id']
	#del train1_data['market']
	#del train2_data['market']
	#del test_data['market']
	del train_data['marketshare_dominant_prod_in_cat']
	del train_data['repeat_buy_prob_90d']
	#del train_data['repeat_buy_prob_60d']
	#del test_data['repeat_buy_prob_60d']
	del train_data['repeat_buy_prob_30d']
	del train_data['prodid_spend_corr']
	del train_data['avg_price_per_cheapest_common_unit']
	del train_data['price_mean_compare']
	del train_data['has_bought_brand_a_60']
	del train_data['prodid_spend_all']
	del train_data['chain']
	del train_data['id']
	del train_data['seasonal_spend_rate_30d']
	del train_data['share_of_cust_bought_prod']
	del train_data['price_quantile']
	del train_data['price_median_compare']
	del train_data['avg_price_per_most_common_unit']
	
	del train_data['established_product']
	del train_data['probability_of_60d_buy_in_category']
	del train_data['num_distinct_products_in_cat_bought']
	del train_data['only_bought_our_product']
	
	del train_data["never_bought_category"]
	del train_data["never_bought_brand"]
	del train_data["never_bought_company"]
	

	del train_data['marketshare_in_dep']
	#del train_data['share_of_cust_bought_dep']
	#del test_data['share_of_cust_bought_dep']
	del train_data['share_dep_spend']
	

	sample_weights = np.ones((train_data.shape[0],1))
	for ofew in offer_est_weights:
		sample_weights[(train_data['offer_id'] == ofew).values,0] = 5.
	
	del train_data['offer_id']
	

	scores = []
	cv = KFold(n=len(train_data), n_folds=10, indices=True)
	for split_index, (train, test) in enumerate(cv):
		print ("# Cross Validation Iteration #%d" % split_index)
		print ("train indices: {0}...".format(train[:10]))
		print ("test indices: {0}...".format(test[:10]))

	for split_index, (train, test) in enumerate(cv):
		count=0
		X_train, y_train = train_data.iloc[train,:], train_label[train]
		X_test, y_test = train_data.iloc[test,:], train_label[test]
		print "hello"
		pprint.pprint(X_train.shape)
		pprint.pprint(y_train.shape)
		dtrain1 = xgb.DMatrix( X_train.values, label=y_train.values, weight=sample_weights[train,0])
		dtrain2 = xgb.DMatrix( X_test.values)
		print "hi"
		#pprint.pprint(dtrain1.shape)
		#pprint.pprint(dtrain2.shape)
		print ("train indices: {0}...".format(train[:10]))
		print ("test indices: {0}...".format(test[:10]))
		param = {'bst:max_depth':5, 'bst:eta':0.3, 'silent':1, 'objective':'binary:logistic', 'nthread' : 4, 'eval_metric':'auc' }
		num_round = 150
		bst = xgb.train( param, dtrain1, num_round)
		pred_label1=bst.predict( dtrain2 )
		pprint.pprint(pred_label1[:10])
		pred_label1[pred_label1>=0.5]=1
		pred_label1[pred_label1<0.5]=0
		pprint.pprint(pred_label1[:10])
		for i in range(len(pred_label1)):
			if pred_label1[i]==y_test.values[i]:
				count=count+1
		print count
		acc=(100*count)/len(y_test)
		print acc
		scores.append(acc)

a=len(scores)
print a
print "Mean(scores)=%.5f\tStddev(scores)=%.5f"%(np.mean(scores), np.std(scores))
print "Done"