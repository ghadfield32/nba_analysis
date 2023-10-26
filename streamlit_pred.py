#streamlit run "c:/Users/ghadf/OneDrive/Desktop/Data Analytics/Python/ML/nba_w_l_prediction_models/nba_analysis/streamlit_pred.py"
#improvements:
# 1. add a button to update the model
# 2. rearrage the order of the upadted Results:
# 3. add dates to the scatter plot so we can see how each model does over time or make line chart
# 4. change matchup_id unique to only take data that the model is sure about so if there's duplicate 0's per matchup_id, drop it
# 5. Legend for Statistics that go into model
# 6. add the basketball chatbot to the streamlit app but make it so they have to input their own api key to use it until we can use it on cv application

#***essentials***
#add filter so predictions date_prediction_recorded can't be later than a games time, so we can't predict after the game has been played
#update correct vs incorrect predictions so it only does the prediction if prediction not null

#Model Improvements:
# add in preprocessing for non-tree classifiers X, now save it and the one_hot_encoder for non-tree's and implement in here for non-tree models
# 1's: SVM, Logistic Regression, ridge classifier, sgd classifier are all very similar in accuracy.  We should try to combine them into one model
# 0's: gaussianNB, KNN, 
# 1's/0's: RandomF, XGBoost, AdaBoost, GradientBoost, DecisionTree

# add in linear regression models for how each team will do for each statistic, maybe even change the averages to these predicted stats if they are good enough
# add in CNN for image recognition of the players
# add in LSTM for time series analysis of how each team does over time
# add in a chatbot that will give you the prediction for the game you ask it about
# add CNN to predict who will win the championship based on the players on the team


#unseen data will only pull in data until tomorrow
# 23_24_season_games_past.csv will provide the actual results for the games that have already been played

import streamlit as st
import pandas as pd

# File paths and feature names
tree_pred_path = 'data/tree_season_pred.csv'
non_tree_pred_path = 'data/non_tree_season_pred.csv'
ltsm_pred_path = 'data/ltsm_season_pred.csv'
past_results_path = 'data/nba_prepreprocess_data.csv'
votes_data_path = 'data/voter_pred.csv'

# Data Loading Functions
def load_tree_data(path):
    return pd.read_csv(path)[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'XGBoost_PREDICTION', 'Decision Tree_PREDICTION', 'Random Forest_PREDICTION', 'Gradient Boosting_PREDICTION', 'AdaBoost_PREDICTION']]

def load_non_tree_data(path):
    return pd.read_csv(path)[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'MLP Classifier_PREDICTION', 'K-Neighbors Classifier_PREDICTION', 'SVM_PREDICTION', 'SGD Classifier_PREDICTION', 'Ridge Classifier_PREDICTION', 'Logistic Regression_PREDICTION']]

def load_ltsm_data(path):
    ltsm_data = pd.read_csv(path)
    ltsm_data = ltsm_data.rename(columns={'PREDICTION': 'ltsm_PREDICTION'})
    return ltsm_data[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'ltsm_PREDICTION']]

def load_past_results(path):
    return pd.read_csv(path)

# Load voter data and remove the index column if it exists
try:
    votes_data = pd.read_csv(votes_data_path).drop(columns=['Unnamed: 0'], errors='ignore')
except FileNotFoundError:
    votes_data = pd.DataFrame(columns=['Date', 'MATCHUP_ID', 'TEAM_NAME', 'Votes'])

# Aggregate votes by Date, MATCHUP_ID, and TEAM_NAME
aggregated_votes = votes_data.groupby(['Date', 'MATCHUP_ID', 'TEAM_NAME']).sum().reset_index().rename(columns={'Votes': 'voter_predictions'})
#print(aggregated_votes.head())

# Merging function
def merge_data(tree_data, non_tree_data, ltsm_data):
    tree_non_tree = pd.merge(tree_data, non_tree_data, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')
    all_data = pd.merge(tree_non_tree, ltsm_data, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')
    return pd.merge(all_data, aggregated_votes, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')

def main():
    st.title("Man vs Machine: NBA Predictions")
    
    # Load and merge data
    tree_data = load_tree_data(tree_pred_path)
    non_tree_data = load_non_tree_data(non_tree_pred_path)
    ltsm_data = load_ltsm_data(ltsm_pred_path)
    all_data = merge_data(tree_data, non_tree_data, ltsm_data)
    past_results = load_past_results(past_results_path)
    
    # Sidebar for navigation
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Man vs Machine", "All Predictions", "Past Results"])
    
    if app_mode == "Man vs Machine":
        st.subheader('Upcoming NBA Games (LSTM Predictions)')
        
        # Load previous votes if available
        if 'votes_data' not in st.session_state:
            try:
                st.session_state['votes_data'] = pd.read_csv(votes_data_path)
            except FileNotFoundError:
                st.session_state['votes_data'] = pd.DataFrame(columns=['MATCHUP_ID', 'TEAM_NAME', 'Votes'])
        
        selected_date = st.selectbox('Select Date for Matchups:', ltsm_data['Date'].unique())
        matchups_on_selected_date = ltsm_data[ltsm_data['Date'] == selected_date]

        # Group by matchup_id to show both teams in the same section
        grouped = matchups_on_selected_date.groupby('MATCHUP_ID')

        for matchup_id, group in grouped:
            for _, matchup in group.iterrows():
                st.write(f"{matchup['TEAM_NAME']}: LSTM: {matchup['ltsm_PREDICTION']}")

            # Voting
            selected_team = st.selectbox(f"Who will win {matchup_id}?", group['TEAM_NAME'].tolist())

            if st.button(f"Vote for {selected_team}"):
                if len(st.session_state['votes_data'].loc[(st.session_state['votes_data']['MATCHUP_ID'] == matchup_id) & (st.session_state['votes_data']['TEAM_NAME'] == selected_team)]) == 0:
                    new_row = {'Date': selected_date, 'MATCHUP_ID': matchup_id, 'TEAM_NAME': selected_team, 'Votes': 1}
                    st.session_state['votes_data'] = pd.concat([st.session_state['votes_data'], pd.DataFrame([new_row])], ignore_index=True)
                else:
                    st.session_state['votes_data'].loc[(st.session_state['votes_data']['MATCHUP_ID'] == matchup_id) & (st.session_state['votes_data']['TEAM_NAME'] == selected_team) & (st.session_state['votes_data']['Date'] == selected_date), 'Votes'] += 1

                st.session_state['votes_data'].to_csv(votes_data_path, index=False)  # Save to CSV

            st.write("---")
    
    elif app_mode == "All Predictions":
        st.subheader('All Predictions')
        st.write(all_data)

    elif app_mode == "Past Results":
        st.subheader('Past NBA Games and Predictions')
        #rename GAME_DATE column in past_results to Date
        past_results = past_results.rename(columns={'GAME_DATE': 'Date'})

        # Merge past results with predictions
        past_data_with_predictions = pd.merge(past_results, all_data, on=['Date', 'TEAM_NAME'], how='left')
        
        # Show past data
        st.write(past_data_with_predictions)
        
if __name__ == "__main__":
    main()










