This Readme file provides an overview on all files in folder /IMPRO_model_results_matlab_code
Author: Uwe Ehret, uwe.ehret@kit.edu
Created: 2025/02/18

MAIN FOLDER

	calculate_results_Hc.m
		- Matlab code to calculate and plot results from all models for objective function conditional entropy (Hc)
		
	calculate_results_KGE.m
		- Matlab code to calculate and plot results from all models for objective function Kling-Gupta efficiency (KGE)
	
	f_KGE.m
		- Matlab function to calculate Kling-Gupta efficiency. Called by calculate_results_KGE.m
	
	f_binme.m
		- Matlab function to classify data into bins
	
	f_conditionalentropy.m
		- Matlab function to calculate the (joint) conditional entropy of an 1-to-any-dimensional discrete (binned) frequency distribution of target variable(s) given prior knowledge of 1-to-any number of discrete (binned) predictor values
	
	f_entropy.m
		- Matlab function to calculate the (joint) entropy of an 1-to-any-dimensional discrete (binned) frequency distribution

	dp_training_points.mat
		- Matlab data set. For each test catchment (Iller, Saale, Selke), contains time positions of values selected for model training by the Douglas-Peucker algorithm
	
	sample_sizes.mat
		- Matlab data set. For each sample size and sampling repetition, contains time positions of values selected for model training

	iller_data.mat, saale_data.mat, selke_data.mat
		- Matlab data sets. For each test catchment, contains spatially distributed observed time series of atmospheric forcing (P, ETP, T) and streamflow (Q) and related timestamps (DateTime). Also, contains subcatchment indices
	
	iller_predictors.mat, saale_predictors.mat, selke_predictors.mat
		- Matlab data sets. For each test catchment, contains time series of predictors, one set for each empirical discrete distribution models (EDDIS models)
		- Sets are ["Ps1t1","Ps1t2","Ps2t1","Ps2t2","Ps1t1Ts1t1","Ps1t2Ts1t1","Ps1t2Ts1t3"]
		- P Precipitation, T Temperature
		- s1 All grid values aggregated to one, s2 grid values aggegrated in two sub-basins
		- t1 only t=0, t2 values for t=0, t=-1, t=-2..-6, t=-7..-30, t3 values for t=0, t=-1..-30

	CONCEPTUAL_MODELS_results.mat, EDDIS_results.mat, LSTM_ANN_results.mat, RTREE_results.mat
		- Matlab data sets. One for each set of model types (CONCEPTUAL, EDDIS, LSTM ANN, RTREE). Each file contains simulated timeseries for each training sample size and each sampling repetition
		

FOLDER /Hc
	- Matlab data sets. Each file contains conditional entropy (Hc) for 9 training sample sizes [10 50 100 250 500 1000 2000 3000 3652] and 30 sampling repetitions.
	- one file for each catchment and model, and (if applicable) sampling scheme, e.g. KGE_iller_RTREE, or KGE_iller_HBV_lumped


FOLDER /KGE
	- Matlab data sets. Each file contains Kling-Gupta efficiency (KGE) for 9 training sample sizes [10 50 100 250 500 1000 2000 3000 3652] and 30 sampling repetitions.
	- one file for each catchment and model, and (if applicable) sampling scheme, e.g. KGE_iller_RTREE, or KGE_iller_HBV_lumped
