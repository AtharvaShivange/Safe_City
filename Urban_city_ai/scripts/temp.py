# import pandas as pd

# df = pd.read_csv("../data/raw/safecity_full.csv")

# print("Total incidents:", len(df))
# print(df.head())
# # print(df.describe())
# df_density = df.groupby(['latitude', 'longitude']).size().reset_index(name='crime_count')

# print(df_density.sort_values(by='crime_count', ascending=False).head(10))

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/raw/safecity_full.csv")

# plt.figure(figsize=(8,8))
# plt.scatter(df['longitude'], df['latitude'], s=1)
# plt.title("Raw Crime Points")
# plt.show()

plt.figure(figsize=(8,8))
plt.hexbin(df['longitude'], df['latitude'], gridsize=50)
plt.title("Crime Density (Hexbin)")
plt.colorbar()
plt.show()