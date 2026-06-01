import pandas as pd

df = pd.read_csv("data/IPL.csv", low_memory=False)

print("Shape:", df.shape)

print("\nInnings:")
print(df['innings'].value_counts())

print("\nUnique Matches:")
print(df['match_id'].nunique())

print("\nBatting Teams:")
print(sorted(df['batting_team'].dropna().unique()))