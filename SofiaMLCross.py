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
	ss = StandardScaler()
	cv = KFold(n=len(train_data), n_folds=10, indices=True)
	for split_index, (train, test) in enumerate(cv):
		print ("# Cross Validation Iteration #%d" % split_index)
		print ("train indices: {0}...".format(train[:10]))
		print ("test indices: {0}...".format(test[:10]))

	for split_index, (train, test) in enumerate(cv):
		count=0
		train_data_norm = ss.fit_transform(train_data.iloc[train,:])
		test_data_norm = ss.transform(train_data.iloc[test,:])
		print "hello"
		pprint.pprint(train_data_norm.shape)
		pprint.pprint(test_data_norm.shape)
		model_file = os.path.join(data_path,"sofml.model")
		training_file = os.path.join(data_path, "train_data.dat")
		test_file = os.path.join(data_path, "test_data.dat")
		pred_test_file = os.path.join(data_path, "pred_test.csv")
		print "writing out files"	
		ntrain_label = train_label[train].copy()
		y_test=train_label[test]
		ntrain_label.values[np.where(ntrain_label == 0)] = -1
		dump_svmlight_file(train_data_norm, ntrain_label, training_file, zero_based=False)
		dump_svmlight_file(test_data_norm, np.zeros((test_data_norm.shape[0],)), test_file, zero_based=False)
		print "hi"
		call(sofiaml_path+" --learner_type sgd-svm --loop_type roc --prediction_type logistic --iterations 200000 --lambda 10000 --training_file "+training_file+" --model_out "+model_file, shell=True)
		call(sofiaml_path+" --model_in "+model_file+" --test_file "+test_file+" --results_file "+pred_test_file, shell=True)
		pred_label_test4 = pd.io.parsers.read_csv(pred_test_file, sep="\t", names=["pred","true"])['pred']
		pprint.pprint(pred_label_test4[:10])
		pred_label_test4= 1./(1.+np.exp(-pred_label_test4))
		pred_label_test4[pred_label_test4>=0.5]=1
		pred_label_test4[pred_label_test4<0.5]=0
		pprint.pprint(pred_label_test4[:10])
		for i in range(len(pred_label_test4)):
			if pred_label_test4[i]==y_test.values[i]:
				count=count+1
		print count
		acc=(100*count)/len(y_test)
		print acc
		scores.append(acc)

a=len(scores)
print a
print "Mean(scores)=%.5f\tStddev(scores)=%.5f"%(np.mean(scores), np.std(scores))
print "Done"