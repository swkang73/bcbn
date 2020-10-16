import numpy as np, pandas as pd

"""
Notes on Analysis:
- we have mean, se, & worst on radius, texture, perimeter, area, smoothness, compactness,
concavity, concave points, symmetry, fractal dimensions 

1st preprocessing: normalize each columns using z-score 
"""



print("Import data")
df = pd.read_csv("data.csv")

print("Yield pairwise attributes")
raw_attr = df.columns.to_list()
attr_set = set([attr.split("_")[0] for attr in raw_attr])

print("calculate global mean")
attr_global_avg = {}
for attr in attr_set:
	val = np.mean(df[attr + "_mean"])
	print("for " + attr + " mean: {:.4f}".format(val))
	attr_global_avg[attr] = val
