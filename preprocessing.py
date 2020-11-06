import numpy as np, pandas as pd
from scipy import stats

"""
Notes on Analysis:
- we have mean, se, & worst on radius, texture, perimeter, area, smoothness, compactness,
concavity, concave points, symmetry, fractal dimensions 

1st preprocessing: normalize each columns using z-score 
"""


print("Import data")
df = pd.read_csv("data.csv")

print("get z-score for each")
new_data = {
	'id': df['id'],
	'diagnosis': df['diagnosis']
}

print("create new attributes by z-score")
features = set(df.columns.to_list()) - set(["id", "diagnosis"])
for f in features:
	if '_se' not in f:
		zscores = stats.zscore(df[f])
		new_data[f] = np.array(zscores > 0, dtype=int)


new_data['diagnosis'] = [int(v=="M") for v in df['diagnosis']] 

print("export processed data")
ndf = pd.DataFrame.from_dict(new_data)
ndf.to_csv("pdata.csv", index=False)

