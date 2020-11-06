import sys, csv
import random
import pandas as pd
import numpy as np
import seaborn as sn

import itertools as it
import time
import networkx as nx
from matplotlib import pyplot as plt
import scipy.special as sc
import math

import scipy.sparse as sps
from collections import defaultdict

from pomegranate import *				# jliu99 Note: Activate cs238 venv to use ---> source cs238/bin/activate


def naive_bayes(X):
	"""
	Takes the data to fit the structure too, where each 
    row is a sample and each column corresponds to the associated variable.
    Builds a naive bayes network, calculates probabilities according to MLE,
    and returns the resulting network.

	:param X: numpy array, shape (n_samples, n_nodes)
	:returns model: Pomegranate Bayesian network object

	"""
	# NODES IN NETWORK:
	# s0 = State( diagnosis, name="diagnosis" )
	# s1 = State( symmetry_se, name="symmetry_se" )
	# s2 = State( compactness_worst, name="compactness_worst" )
	# s3 = State( radius_worst, name="radius_worst" )
	# s4 = State( smoothness_worst, name="smoothness_worst" )
	# s5 = State( radius_mean, name="radius_mean" )
	# s6 = State( concavity_mean, name="concavity_mean" )
	# s7 = State( concavePoints_worst, name="concavePoints_worst" )
	# s8 = State( texture_se, name="texture_se" )
	# s9 = State( area_worst, name="area_worst" )
	# s10 = State( area_se, name="area_se" )
	# s11 = State( concavePoints_se, name="concavePoints_se" )
	# s12 = State( compactness_se, name="compactness_se" )
	# s13 = State( texture_mean, name="texture_mean" )
	# s14 = State( symmetry_worst, name="symmetry_worst" )
	# s15 = State( compactness_mean, name="compactness_mean" )
	# s16 = State( concavity_se, name="concavity_se" )
	# s17 = State( perimeter_se, name="perimeter_se" )
	# s18 = State( fractalDimension_mean, name="fractalDimension_mean" )
	# s19 = State( concavePoints_mean, name="concavePoints_mean" )
	# s20 = State( smoothness_se, name="smoothness_se" )
	# s21 = State( smoothness_mean, name="smoothness_mean" )
	# s22 = State( concavity_worst, name="concavity_worst" )
	# s23 = State( symmetry_mean, name="symmetry_mean" )
	# s24 = State( radius_se, name="radius_se" )
	# s25 = State( perimeter_worst, name="perimeter_worst" )
	# s26 = State( area_mean, name="area_mean" )
	# s27 = State( fractalDimension_worst, name="fractalDimension_worst" )
	# s28 = State( texture_worst, name="texture_worst" )
	# s29 = State( fractalDimension_se, name="fractalDimension_se" )
	# s30 = State( perimeter_mean, name="perimeter_mean" )


	network = BayesianNetwork( "Breast Cancer Biopsies" )

	# Add nodes to the network
	for col in range(X.shape[1]):
		network.add_node(col)

	diagnosis_state = [0]
	symptom_states = [i for i in range(1, X.shape[1])]

	# BUILD MODEL USING FROM_SAMPLES FUNCTION WITH SPEED UP ARGUMENTS.
	# ----------------------------------------------------------------------
	# Add edges to the network
	for edge in list(it.product(symptom_states, diagnosis_state)):
		network.add_edge(edge[0], edge[1])

	# Fit data to Naive Bayes structure
	# Algorithm options: 'chow-liu' (fast), ‘greedy’, ‘exact’, ‘exact-dp’ 
	ALGORITHM = 'greedy'		
	# Root options: For algorithms which require a single root (‘chow-liu’), this is the root for which all edges point away from. 
	# 				User may specify which column to use as the root. Default is the first column.
	ROOT = 2
	# n_jobs options: The number of threads to use when learning the structure. Will parallelize if constraint graph provided.
	N_JOBS = 5
	model = network.from_samples(X, algorithm=ALGORITHM, constraint_graph=network, name="Breast Cancer Biopsy Model", n_jobs=N_JOBS)


	# BUILD MODEL USING FROM_STRUCTURE FUNCTION --- SEEMS SLOW.
	# ----------------------------------------------------------------------
	# Define parents for each node:
	# Naive Bayes structure: node 0 (diagnosis) has 30 parents, other nodes
	# have no parents.
	# parents = (tuple(symptom_states), )
	# for state in symptom_states:
	# 	parents = parents + ((), )

	# # Fit data to Naive Bayes structure
	# model = network.from_structure(X, structure=parents, name="Breast Cancer Naive Bayesian")
	

	return model


def struct_learning(X, root=2):
	"""
	Takes the data to fit the structure too, where each 
    row is a sample and each column corresponds to the associated variable.
    Constructs a probable network
    and returns the resulting network.

	:param X: numpy array, shape (n_samples, n_nodes)
	:returns model: Pomegranate Bayesian network object

	"""
	network = BayesianNetwork( "Breast Cancer Biopsies" )
	# Learn structure from data.
	# Algorithm options: 'chow-liu' (fast), ‘greedy’, ‘exact’, ‘exact-dp’
	ALGORITHM = 'chow-liu'	 
	# Max parents options: Set to a lower number if you want to speed up learning. (Default: total number of nodes - 1)
	MAX_PARENTS = X.shape[1] - 1		
	#MAX_PARENTS = 1
	# Root options: For algorithms which require a single root (‘chow-liu’), this is the root for which all edges point away from. 
	# 				User may specify which column to use as the root. Default is the first column.
	ROOT = root
	print(ROOT)
	model = network.from_samples(X, algorithm=ALGORITHM, max_parents=MAX_PARENTS, root=ROOT, name="Breast Cancer Biopsy Model")
	return model


def graph_plot(network, pltname = 'temp_struct.png'):
	"""
	Plots the network found by the model as a directed graph
	Requires user to install matplolib and pygraphviz

	pygraphviz install commands (if you're having trouble):
	
	brew install graphviz
	pip install --install-option="--include-path=/usr/local/include/" --install-option="--library-path=/usr/local/lib/" pygraphviz

	:param network: Pomegranate Bayesian network model
	"""
	plt.figure(figsize=(16, 8))
	network.plot()
	plt.savefig(pltname)
	plt.clf()


def score_learning_strategy(predicted, true, cf_name='temp.png'):
	"""
	Outputs a confusion matrix to evaluate the results of a machine learning classification model.

	:param predicted: List of classifications as predicted by the model
	:param true: List of true classifications as indicated by the data
	"""
	print(predicted)
	print(true)

	data = {'y_Actual': true, 'y_Predicted': predicted}

	df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
	confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
	print (confusion_matrix)

	# Uncomment these lines if you want to print a confusion matrix png.
	sn.set(font_scale=1.4) # for label size
	sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 16}) # font size
	plt.savefig(cf_name)
	plt.clf()

	# Since this is a disease diagnosis prediction problem, our measure of performance 
	# will be RECALL = TP/(TP + FN)
	tp = confusion_matrix['M']['M']
	fn = confusion_matrix['M']['B']

	return tp / (tp + fn)


def split(X):
	benign = [X[i] for i in range(X.shape[0]) if X[i][0] == 'B']
	malignant = [X[i] for i in range(X.shape[0]) if X[i][0] == 'M']

	benign_test_idx = list(random.sample(range(len(benign)), k=35))
	malignant_test_idx = list(random.sample(range(len(malignant)), k=35))
	benign_train_idx = [i for i in range(len(benign)) if i not in benign_test_idx]
	malignant_train_idx = [i for i in range(len(malignant)) if i not in malignant_test_idx]

	benign_test_set = [benign[i] for i in benign_test_idx]
	malignant_test_set = [malignant[i] for i in malignant_test_idx]
	benign_train_set = [benign[i] for i in benign_train_idx]
	malignant_train_set = [malignant[i] for i in malignant_train_idx]

	test = np.row_stack((benign_test_set, malignant_test_set))
	train = np.row_stack((benign_train_set, malignant_train_set))

	return train, test


def build_model(X, root, cf_name):

	# data = pd.read_csv('p_data.csv').to_numpy()
	# X = data[:, 1:]

	# # Remove 70 values from dataset to use as test set.
	# train, test = split(X)

	# Keep track of these values for accuracy metrics.
	accuracies = list()
	true_diagnoses = list(X[:, 0])
	predicted_diagnoses = list()

	# Due to dataset size, we will perform leave one out cross validation:
	for r in range(X.shape[0]):

		print("Building model %s... " %r)

		train_loocv = np.copy(X) 
		train_loocv = np.delete(train_loocv, r, axis=0)
		valid = np.copy(X[r,:]).reshape((1, X.shape[1]))
		actual_diagnosis = valid[0, 0]

		# Remove the actual diagnosis from the test sample
		valid[0, 0] = np.nan

		# Option 1: Naive Bayes
		#network = naive_bayes(train_loocv, root)

		# Option 2: Structure learning
		network = struct_learning(train_loocv, root)

		if (r == X.shape[0] - 1):

			# Plot learned network
			graph_plot(network, pltname = 'struct' + str(root) + '.png')

		# Used learned network to make a prediction
		predicted_diagnosis = network.predict(valid)[0][0]
		predicted_diagnoses.append(predicted_diagnosis)

	# Score final learning strategy on held-out validation sets.
	return score_learning_strategy(predicted_diagnoses, true_diagnoses, cf_name)



def main():

	t0 = time.time()

	# Read in data
	data = pd.read_csv('p_data.csv').to_numpy()
	X = data[:, 1:]

	# Remove 70 values from dataset to use as test set.
	train, test = split(X)

	# Determine which node is the best "root" using the validation set
	# (For stucture learning with the Chow-Liu algorithm.)

	best_root = 0
	best_score = 0
	NUM_ROOTS = 31

	recalls = list()
	roots = range(NUM_ROOTS)

	for n in range(1, NUM_ROOTS):
		print ("Evaluating node %s as root..." %n)
		score = build_model(train, root = n, cf_name = 'root' + str(n) +'.png')
		recalls.append(score)
		if score > best_score: 
			best_score = score
			best_root = n

	# Construct plot of roots and model performance.
	plt.plot(roots, recalls, 'o', color='black');
	plt.xlabel('Root')
	plt.ylabel('Recall')
	plt.savefig('root_optimization.png')
	plt.clf()

	# FINAL STEP: -------------> DO NOT UNCOMMENT AND RUN UNTIL LAST STEP
	# After optimizing hyperparameters on validation set, build 
	# a final network using full training set and test on full test set. 
	
	final_network = struct_learning(train, root=best_root)
	true_diagnoses = np.copy(test[:, 0])
	test[:, 0] = np.nan
	predicted_diagnoses = [p[0] for p in final_network.predict(test)]
	recall = score_learning_strategy(predicted_diagnoses, true_diagnoses, cf_name = 'final.png')
	graph_plot(final_network, pltname = 'final_struct.png')
	print("FINAL RECALL: ", recall)
	# ---------------------------------------------------------------------


	t1 = time.time()

	# Print total runtime.
	print("TOTAL RUNTIME: ", t1-t0)


if __name__ == '__main__':
	main()
