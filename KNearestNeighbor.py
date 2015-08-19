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
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.decomposition import PCA


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
		X_train, y_train = train_data.iloc[train,:], train_label[train]
		X_test, y_test = train_data.iloc[test,:], train_label[test]
		print "hello"
		pprint.pprint(X_train.shape)
		pprint.pprint(y_train.shape)
		#X1_new=SelectKBest(f_classif,k=71).fit_transform(X_train,y_train)
		#X2_new=SelectKBest(f_classif,k=71).fit_transform(X_test,y_test)
		print "hi"
		#pprint.pprint(X1_new.shape)
		#pprint.pprint(X2_new.shape)
		print ("train indices: {0}...".format(train[:10]))
		print ("test indices: {0}...".format(test[:10]))
		pca = PCA()
		X_transformed = pca.fit_transform(X_train)
		knn = neighbors.KNeighborsClassifier(n_neighbors=80)
		knn.fit(X_transformed, y_train)
		newdata_transformed = pca.transform(X_test)
		scores.append(knn.score(newdata_transformed, y_test))
		print len(scores)

a=len(scores)
print a
print "Mean(scores)=%.5f\tStddev(scores)=%.5f"%(np.mean(scores), np.std(scores))
print "Done"