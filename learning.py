import sys, csv
import random
import pandas as pd
import numpy as np

import itertools as it
import time
import networkx as nx
from matplotlib import pyplot as plt
import scipy.special as sc
import math

import scipy.sparse as sps
from collections import defaultdict

from pomegranate import *


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


def struct_learning(X):
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
	ROOT = 2
	model = network.from_samples(X, algorithm=ALGORITHM, max_parents=MAX_PARENTS, root=ROOT, name="Breast Cancer Biopsy Model")
	return model


def plot(network):
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
	plt.show()


def score_learning_strategy(predicted, true):

	data = {'y_Actual': true, 'y_Predicted': predicted}

	df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
	confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
	print (confusion_matrix)


def main():

	data = pd.read_csv('small_test.csv').to_numpy()
	X = data[:, 1:]

	# Keep track of these values for accuracy metrics.
	accuracies = list()
	true_diagnoses = list(data[:, 1])
	predicted_diagnoses = list()

	t0 = time.time()

	print("Option 1: Bayesian Model")

	# Due to dataset size, we will perform leave one out cross validation:
	for r in range(X.shape[0]):

		print("Building model %s... " %r)

		train = np.copy(X) 
		train = np.delete(train, r, axis=0)
		test_X = np.copy(X[r,:]).reshape((1, X.shape[1]))
		actual_diagnosis = test_X[0, 0]

		# Remove the actual diagnosis from the test sample
		test_X[0, 0] = np.nan

		# Option 1: Naive Bayes
		network = naive_bayes(train)

		# Option 2: Structure learning
		#network = struct_learning(train)

		# Plot learned network
		#plot(network)

		# Used learned network to make a prediction
		predicted_diagnosis = network.predict(test_X)[0][0]
		predicted_diagnoses.append(predicted_diagnosis)

	# Score final learning strategy
	score_learning_strategy(predicted_diagnoses, true_diagnoses)

	t1 = time.time()

	# Print total runtime.
	#print("TOTAL RUNTIME: ", t1-t0)


if __name__ == '__main__':
	main()
