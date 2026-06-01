import pandas as pd

df = pd.read_csv("data/IPL.csv", low_memory=False)

# Keep only 2nd innings (chases)
df = df[df['innings'] == 2].copy()

# Current score
df['current_score'] = df.groupby('match_id')['runs_total'].cumsum()

# Balls bowled per match
df['balls_bowled'] = df.groupby('match_id').cumcount() + 1

# Runs left
df['runs_left'] = df['runs_target'] - df['current_score']

# Balls left
df['balls_left'] = 120 - df['balls_bowled']

# Wickets left
df['wickets_left'] = 10 - df['team_wicket']

# Current Run Rate
df['current_run_rate'] = (df['current_score'] * 6) / df['balls_bowled']

# Required Run Rate
df['required_run_rate'] = (df['runs_left'] * 6) / df['balls_left']

# Target variable (win/loss)
df['result'] = df['match_won_by'] == df['batting_team']
df['result'] = df['result'].astype(int)

print(df[['runs_left','balls_left','wickets_left','current_run_rate','required_run_rate','result']].head(10))