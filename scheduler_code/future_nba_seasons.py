# %%
import pandas as pd
import numpy as np

# Reading the data from the CSV file
data = pd.read_csv('data\23_24_season_games.csv')

# Split the data into two separate DataFrames for home and away
home_data = data[['DATE', 'Start (ET)', 'Home/Neutral']].copy()
home_data['Home_Away'] = 'Home'
home_data['MATCHUP'] = home_data['Home/Neutral'] + ' vs. ' + data['Visitor/Neutral']
home_data.rename(columns={'Home/Neutral': 'Team'}, inplace=True)
home_data['WL_encoded'] = np.nan  # Placeholder for predictions or actual results

away_data = data[['DATE', 'Start (ET)', 'Visitor/Neutral']].copy()
away_data['Home_Away'] = 'Away'
away_data['MATCHUP'] = home_data['Team'] + ' vs. ' + away_data['Visitor/Neutral']
away_data.rename(columns={'Visitor/Neutral': 'Team'}, inplace=True)
away_data['WL_encoded'] = np.nan  # Placeholder for predictions or actual results

# Union/Concatenate the two DataFrames
final_data = pd.concat([home_data, away_data], ignore_index=True)

# Sorting the data by Date, Start time, and Home/Away status
final_data.sort_values(by=['DATE', 'Start (ET)', 'Home_Away'], inplace=True)

#reset the index
final_data.reset_index(drop=True, inplace=True)
print(final_data)

# Write the data to a CSV file
#final_data.to_csv(r'C:\Users\ghadf\OneDrive\Desktop\Data Analytics\Python\ML\nba_w_l_prediction_models\nba_analysis\data\23_24_season_games.csv', index=False)

# %%
import pandas as pd

# Assuming you've already read in your df1
df1 = pd.read_csv('data\team_ids.csv')

# Filter df1 to only unique TEAM_ID, TEAM_NAME, and SEASON_ID
unique_teams = df1[['TEAM_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates()

#change team name "LA Clippers" to "Los Angeles Clippers"
unique_teams['TEAM_NAME'] = unique_teams['TEAM_NAME'].replace('LA Clippers', 'Los Angeles Clippers')

# get unique values from team_id and season_id columns
unique_team_ids = unique_teams['TEAM_ID'].unique()
print(unique_team_ids)
unique_team_names = unique_teams['TEAM_NAME'].unique()
print(unique_team_names)
unique_team_names = unique_teams['TEAM_ABBREVIATION'].unique()
print(unique_team_names)

#take only the columns team_id, team_name, season_id, and game_id

#print(df1.head())

# Assuming df1 has been read in and unique_teams has been created
team_to_abbreviation = dict(zip(unique_teams['TEAM_NAME'], unique_teams['TEAM_ABBREVIATION']))

print(team_to_abbreviation)


# %% [markdown]
# 1610612737, 1610612738, 1610612739, 1610612740, 1610612741, 1610612742, 1610612743, 1610612744, 1610612745, 1610612746, 1610612747, 1610612748, 1610612749, 1610612750, 1610612751, 1610612752, 1610612753, 1610612754, 1610612755, 1610612756, 1610612757, 1610612758, 1610612759, 1610612760, 1610612761, 1610612762, 1610612763, 1610612764, 1610612765, 1610612766

# %% [markdown]
# current_teams = [1610612739, 1610612737, 1610612738, 1610612740, 1610612741, 1610612742, 1610612743, 1610612744, 1610612745, 1610612746, 1610612747, 1610612748, 1610612749, 1610612750, 1610612751, 1610612752, 1610612753, 1610612754, 1610612755, 1610612756, 1610612757, 1610612758, 1610612759, 1610612760, 1610612761, 1610612762, 1610612763, 1610612764, 1610612765, 1610612766]
# 

# %%


# Convert 'Date' column to datetime format in final_data
final_data['DATE'] = pd.to_datetime(final_data['DATE'])


# Merge final_data with unique_teams on the team names to get the TEAM_ID. Use 'left' to ensure final_data size isn't increased.
final_data = final_data.merge(unique_teams, left_on='Team', right_on='TEAM_NAME', how='left')


#count nan values
final_data.isnull().sum()
#view the team_id columns that have nan values
final_data[final_data['TEAM_ID'].isnull()]

print(final_data)

# %%
# Create a function to extract both teams from the matchup string
def extract_teams(matchup):
    # Split the string using 'vs.' as the delimiter
    teams = matchup.split(' vs. ')
    return teams

# Apply the function to the 'MATCHUP' column
final_data['Home_Team'], final_data['Away_Team'] = zip(*final_data['MATCHUP'].map(extract_teams))

# Determine the opposing team for each row
final_data['Opposing_Team'] = final_data.apply(lambda row: row['Away_Team'] if row['Team'] == row['Home_Team'] else row['Home_Team'], axis=1)

# Create a mapping from the team name to the TEAM_ID using the unique_teams dataframe
team_to_id = dict(zip(unique_teams['TEAM_NAME'], unique_teams['TEAM_ID']))

# Map the opposing team name to its ID
final_data['TEAM_ID_OPP'] = final_data['Opposing_Team'].map(team_to_id)

# Drop the columns we created for the intermediate steps (optional)
final_data.drop(columns=['Home_Team', 'Away_Team', 'Opposing_Team'], inplace=True)
print(final_data)
print(final_data.columns)

# %%


# Convert TEAM_ID and SEASON_ID to integers
#final_data['TEAM_ID'] = final_data['TEAM_ID'].astype(int)
#final_data['SEASON_ID'] = final_data['SEASON_ID'].astype(int)

# If you want to convert them to strings after converting to integers, uncomment the following lines:
# final_data['TEAM_ID'] = final_data['TEAM_ID'].astype(str)
# final_data['SEASON_ID'] = final_data['SEASON_ID'].astype(str)

# Drop 'TEAM_NAME' and 'Start (ET)' column as it's redundant
final_data.drop(['TEAM_NAME', 'Start (ET)'], axis=1, inplace=True)

final_data['TEAM_ID'] = final_data['TEAM_ID'].astype('int64')

# Assuming your dataframe is named 'final_data'
final_data['YEAR'] = pd.to_datetime(final_data['DATE']).dt.year
final_data['MONTH'] = pd.to_datetime(final_data['DATE']).dt.month
final_data['DAY'] = pd.to_datetime(final_data['DATE']).dt.day

# Dropping the original Date column and Team, Home_Away, Matchup columns
final_data.drop(['DATE', 'Team'], axis=1, inplace=True)

# add a matchup identifier column on when they happen on the same day
#final_data['GAME_ID'] = final_data.groupby(['MATCHUP','YEAR', 'MONTH', 'DAY']).ngroup()

print(final_data)


# %%

def replace_team_names_with_abbreviations(row):
    matchup_str = row['MATCHUP']
    for team, abbreviation in team_to_abbreviation.items():
        matchup_str = matchup_str.replace(team, abbreviation)
    
    # If the team is Away, replace "vs." with "@"
    if row['Home_Away'] == 'Away':
        matchup_str = matchup_str.replace(" vs. ", " @ ")
    return matchup_str


final_data['MATCHUP'] = final_data.apply(replace_team_names_with_abbreviations, axis=1)
#print(final_data)
print(len(final_data))
# Create the unique matchup ID
def create_matchup_id(matchup):
    # Split the teams based on " vs. " or " @ "
    teams = matchup.split(' vs. ') if ' vs. ' in matchup else matchup.split(' @ ')
    # Sort the teams alphabetically and concatenate
    return ''.join(sorted(teams))

final_data['MATCHUP_ID'] = final_data['MATCHUP'].apply(create_matchup_id)
print(final_data.head())
print(len(final_data))


# %%
# Taking the weeks worth of games
# Create a Date column in the DataFrame
final_data['Date'] = pd.to_datetime(final_data[['YEAR', 'MONTH', 'DAY']])

#***making into a 1 row per game format***
#get the first matchup_id per date
#final_data = final_data.groupby(['MATCHUP_ID', 'Date']).first().reset_index()


# Get today's date
today = pd.Timestamp.today()-pd.Timedelta(days=1)

# Get the date for one week from today
week_out = today + pd.Timedelta(days=7)

# Filter final_data for dates from today up to one week from now
upcoming_games = final_data[(final_data['Date'] >= today) & (final_data['Date'] <= week_out)] # 
upcoming_games.reset_index(drop=True, inplace=True)
#sort by date
upcoming_games.sort_values(by=['Date'], inplace=True)
print(upcoming_games)

# Filter data for dates before tomorrow and save it as old_data
today = pd.Timestamp.today()+pd.Timedelta(days=1)
old_data = final_data#[final_data['Date'] <= today]

#rename WL_encoded to ACTUAL_RESULT 
old_data.rename(columns={'WL_encoded': 'ACTUAL_RESULT'}, inplace=True)
#add a prediction column full of nan
#old_data['PREDICTION'] = np.nan
#print(old_data)
old_data.to_csv('data\23_24_season_games_past.csv', index=False)

# Drop the 'Date' column
#final_data.drop('Date', axis=1, inplace=True)
#upcoming_games.drop('Date', axis=1, inplace=True)
#old_data.drop('Date', axis=1, inplace=True)

#make predictions on 23_24_season_games_clean (upcoming games with previous season averages)
#saving plan: make predictions on 23_24_season_games_clean (upcoming games with previous season averages)
#1. get the weeks worth of upcoming games from today on so we can't predict on previous games
#2. filter the old data for games including today (so we can add the predictions to this data and the actual results will get added the next day without changing the prediction based on new data)
#3. merge the old data with the predictions data to save yesterday's results with the prediction attached
#4. old_data will have today's games and the predictions will be input onto that data, 
# yesterday's games will have the actual results and will update just that because the model can only predict on today's games moving forward (upcoming_games)
#5. the predictions will be saved to a csv file 

# %%
# Read in the data for averages from X after it's dropped the columns we don't need and just before preprocessing through encoding
#on a long short-term basis

future_season_data_stats = pd.read_csv('data\future_season_data_stats.csv')
#print(future_season_data_stats.head())
#drop WL_encoded column
#future_season_data_stats.drop(['WL_encoded', 'TEAM_ABBREVIATION', 'MATCHUP', 'YEAR', 'MONTH', 'DAY'], axis=1, inplace=True)

print(future_season_data_stats.head())
print(len(future_season_data_stats))



# %%


# Merge final_data with future_season_data_stats on TEAM_ID and Home_Away because we want to see the averages by home and away attached for the models
combined_data = pd.merge(upcoming_games, future_season_data_stats, on=['TEAM_ID', 'Home_Away'], how='left') #, 'MATCHUP'
#print(combined_data.head())

#count nan values
#print(combined_data.isnull().sum())


#view the team_id columns that have nan values
#print(combined_data[combined_data['FG3A'].isnull()])

# drop  Home_Away_x, MATCHUP_x, TEAM_ABBREVIATION_x and change any columns that end with _y to not have _y at the end
#combined_data = combined_data.rename(columns={'Home_Away_y': 'Home_Away', 'MATCHUP_y': 'MATCHUP'})

# drop the team_abbreviaton column
combined_data.drop(['TEAM_ABBREVIATION'], axis=1, inplace=True)
print(combined_data.head())

# matchup unique values
print(combined_data['MATCHUP'].unique())
#drop matchup
combined_data.drop(['MATCHUP'], axis=1, inplace=True)

#print length of combined_data
print(len(combined_data))

feature_order = ['PTS_PER_MIN', 'PTS_DIFF', 'PTS_PER_MIN_DIFF','TEAM_ID', 'TEAM_ID_OPP', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS', 'Home_Away', 'MATCHUP_ID',# 'FG_PCT_OPP', 'FG3_PCT_OPP',  'SEASON_ID', 'GAME_ID'
                   'TS%', 'ORtg', 'PER%', 'eFG%', 'AST%', #'FT_PCT_OPP', 'PLUS_MINUS_OPP','TS%_OPP', 'eFG%_OPP', 'AST%_OPP', , 'MATCHUP'
                  'YEAR', 'MONTH', 'DAY', #'DRtg', 'DPER%',
                 'FG_PCT_DIFF','FG3_PCT_DIFF','FT_PCT_DIFF','TS%_DIFF','eFG%_DIFF','AST%_DIFF','ORtg_DIFF','PER%_DIFF'] #, 'MATCHUP'

categorical_features = [ 'TEAM_ID', 'TEAM_ID_OPP', 'Home_Away', 'MATCHUP_ID'] 

# Reorder columns in the new_data DataFrame
combined_data = combined_data[feature_order]
print(combined_data.head())

#filter for only today's games
today = pd.Timestamp.today()
combined_data = combined_data[combined_data['YEAR'] == today.year]
combined_data = combined_data[combined_data['MONTH'] == today.month]
combined_data = combined_data[combined_data['DAY'] == today.day]
print(combined_data.shape)

# Write the data to a CSV file
combined_data.to_csv('data\23_24_season_games_clean.csv', index=False)


# %%



