import numpy as np
import pandas as pd



df = pd.read_csv("")
missing_markers = ["", " ", "NA", "NaN", "null", "0.0", 0, -3000]
df.replace(missing_markers, np.nan, inplace=True)
date_col = df.iloc[:, 0]
data = df.iloc[:, 1:]
df = data.interpolate(method='cubic')
df = pd.concat([date_col, df], axis=1)
df.to_csv("fill.csv", index=False)
