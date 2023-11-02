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

# File paths and feature names
tree_pred_path = 'data/tree_season_pred.csv'
non_tree_pred_path = 'data/non_tree_season_pred.csv'
ltsm_pred_path = 'data/ltsm_season_pred.csv'
ltsm_seq_pred_path = 'data/ltsm_seq_season_pred.csv'
past_results_path = 'data/nba_threeptera_prepreprocess_data.csv'
votes_data_path = 'data/voter_pred.csv'

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Data Loading Functions
def load_tree_data(path):
    return pd.read_csv(path)[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'XGBoost_PREDICTION', 'Decision Tree_PREDICTION', 'Random Forest_PREDICTION', 'Gradient Boosting_PREDICTION', 'AdaBoost_PREDICTION']]

def load_non_tree_data(path):
    return pd.read_csv(path)[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'MLP Classifier_PREDICTION', 'K-Neighbors Classifier_PREDICTION', 'SVM_PREDICTION', 'SGD Classifier_PREDICTION', 'Ridge Classifier_PREDICTION', 'Logistic Regression_PREDICTION']]

def load_ltsm_data(path):
    ltsm_data = pd.read_csv(path)
    ltsm_data = ltsm_data.rename(columns={'PREDICTION': 'ltsm_PREDICTION'})
    return ltsm_data[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'ltsm_PREDICTION']]

def load_ltsm_seq_data(path):
    ltsm_seq_data = pd.read_csv(path)
    ltsm_seq_data = ltsm_seq_data.rename(columns={'PREDICTION': 'ltsm_seq_PREDICTION'})
    return ltsm_seq_data[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'ltsm_seq_PREDICTION']]

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
def merge_data(tree_data, non_tree_data, ltsm_seq_data, ltsm_data):
    tree_non_tree = pd.merge(tree_data, non_tree_data, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')
    tree_non_tree_ltsm = pd.merge(tree_non_tree, ltsm_seq_data, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')
    all_data = pd.merge(tree_non_tree_ltsm, ltsm_data, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')
    return pd.merge(all_data, aggregated_votes, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')

def calculate_daily_accuracy(data):
    data['correct_prediction'] = data['ltsm_seq_PREDICTION'] == data['WL_encoded']
    daily_accuracy = data.groupby('Date')['correct_prediction'].mean().reset_index()
    daily_accuracy['Date'] = pd.to_datetime(daily_accuracy['Date'])
    return daily_accuracy.sort_values(by='Date')

def main():
    st.title("Man vs Machine: NBA Predictions")
    
    # Load and merge data
    tree_data = load_tree_data(tree_pred_path)
    non_tree_data = load_non_tree_data(non_tree_pred_path)
    ltsm_data = load_ltsm_data(ltsm_pred_path)
    ltsm_seq_data = load_ltsm_seq_data(ltsm_seq_pred_path)
    all_data = merge_data(tree_data, non_tree_data, ltsm_data, ltsm_seq_data)
    past_results = load_past_results(past_results_path)
    
    # Sidebar for navigation
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Man vs Machine", "All Predictions", "Voter Predictions"])
    
    if app_mode == "Voter Predictions":
        st.subheader('Upcoming NBA Games (LSTM Predictions)')
        
        # Load previous votes if available
        if 'votes_data' not in st.session_state:
            try:
                st.session_state['votes_data'] = pd.read_csv(votes_data_path)
            except FileNotFoundError:
                st.session_state['votes_data'] = pd.DataFrame(columns=['MATCHUP_ID', 'TEAM_NAME', 'Votes'])
        
        today_date = datetime.now().strftime('%Y-%m-%d')  # Get today's date
        selected_date = today_date  # Allow voting only for today's games
        matchups_on_selected_date = ltsm_data[ltsm_data['Date'] == selected_date]

        # Group by matchup_id to show both teams in the same section
        grouped = matchups_on_selected_date.groupby('MATCHUP_ID')

        for matchup_id, group in grouped:
            for _, matchup in group.iterrows():
                st.write(f"{matchup['TEAM_NAME']}: LSTM: {matchup['ltsm_PREDICTION']}")

            # Voting
            selected_team = st.selectbox(f"Who will win {matchup_id}?", group['TEAM_NAME'].tolist())

            if st.button(f"Vote for {selected_team}"):
                for team_name in group['TEAM_NAME'].tolist():
                    new_row = {'Date': selected_date, 'MATCHUP_ID': matchup_id, 'TEAM_NAME': team_name, 'Votes': 1 if team_name == selected_team else 0}
                    
                    if len(st.session_state['votes_data'].loc[(st.session_state['votes_data']['MATCHUP_ID'] == matchup_id) & (st.session_state['votes_data']['TEAM_NAME'] == team_name) & (st.session_state['votes_data']['Date'] == selected_date)]) == 0:
                        st.session_state['votes_data'] = pd.concat([st.session_state['votes_data'], pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        st.session_state['votes_data'].loc[(st.session_state['votes_data']['MATCHUP_ID'] == matchup_id) & (st.session_state['votes_data']['TEAM_NAME'] == team_name) & (st.session_state['votes_data']['Date'] == selected_date), 'Votes'] += new_row['Votes']

                st.session_state['votes_data'].to_csv(votes_data_path, index=False)  # Save to CSV

            st.write("---")
    
    elif app_mode == "All Predictions":
        st.subheader('All Predictions')
        #sort by date and reset index
        all_data = all_data.sort_values(by=['Date'], ascending=False)
        all_data = all_data.reset_index(drop=True)
        st.write(all_data)

    elif app_mode == "Man vs Machine":
        st.subheader('Past NBA Games and Predictions')
        past_results = past_results.rename(columns={'GAME_DATE': 'Date'})
        
        # Merge past results with predictions
        past_data_with_predictions = pd.merge(past_results, all_data, on=['Date', 'TEAM_NAME'], how='left')

        # Filter and sort data
        past_data_with_predictions = past_data_with_predictions[past_data_with_predictions['ltsm_PREDICTION'].notna()]
        past_data_with_predictions = past_data_with_predictions.sort_values(by=['Date', 'MATCHUP_ID_x'], ascending=False)
        past_data_with_predictions = past_data_with_predictions.reset_index(drop=True)

        # Only include the columns we want
        past_data_with_predictions = past_data_with_predictions[['Date', 'MATCHUP', 'TEAM_NAME', 'WL', 'WL_encoded', 'ltsm_PREDICTION', 
                                                                 'ltsm_seq_PREDICTION', 'voter_predictions']]
        #print(past_data_with_predictions.head())

        # Calculate daily accuracy for each prediction model
        def calculate_daily_accuracy(data, prediction_column):
            correct_predictions_column = f'{prediction_column}_correct'
            data[correct_predictions_column] = data[prediction_column] == data['WL_encoded']
            daily_accuracy = data.groupby('Date')[correct_predictions_column].mean().reset_index()
            daily_accuracy['Date'] = pd.to_datetime(daily_accuracy['Date'])
            daily_accuracy = daily_accuracy.sort_values(by='Date')
            daily_accuracy['Model'] = prediction_column
            daily_accuracy['correct_prediction'] = daily_accuracy[correct_predictions_column] * 100  # Convert to percentage
            return daily_accuracy

        lstm_accuracy = calculate_daily_accuracy(past_data_with_predictions, 'ltsm_PREDICTION')
        lstm_seq_accuracy = calculate_daily_accuracy(past_data_with_predictions, 'ltsm_seq_PREDICTION')
        voter_accuracy = calculate_daily_accuracy(past_data_with_predictions, 'voter_predictions')

        # Combine accuracies into a single DataFrame
        all_accuracies = pd.concat([lstm_accuracy, lstm_seq_accuracy, voter_accuracy])

        # Filter for the last month's data
        one_month_ago = datetime.now() - pd.to_timedelta(30, unit='d')
        all_accuracies = all_accuracies[all_accuracies['Date'] > one_month_ago]

        # Create line chart
        fig = px.line(all_accuracies, x='Date', y='correct_prediction', color='Model', title='Correct Predictions Over Time', labels={'correct_prediction': 'Accuracy (%)'})
        fig.update_yaxes(tickvals=[i for i in range(0, 101, 10)], ticktext=[f'{i}%' for i in range(0, 101, 10)])
        st.plotly_chart(fig)


        # Calculate accuracy for LSTM predictions
        correct_lstm = sum(past_data_with_predictions['ltsm_PREDICTION'] == past_data_with_predictions['WL_encoded'])
        total_lstm = len(past_data_with_predictions['ltsm_PREDICTION'].dropna())
        accuracy_lstm = round((correct_lstm / total_lstm * 100) if total_lstm != 0 else 0, 2)

        # Calculate accuracy for LSTM sequence predictions
        correct_lstm_seq = sum(past_data_with_predictions['ltsm_seq_PREDICTION'] == past_data_with_predictions['WL_encoded'])
        total_lstm_seq = len(past_data_with_predictions['ltsm_seq_PREDICTION'].dropna())
        accuracy_lstm_seq = round((correct_lstm_seq / total_lstm_seq * 100) if total_lstm_seq != 0 else 0, 2)

        # Handle voter predictions (-1's to 0's) and calculate accuracy
        past_data_with_predictions['voter_predictions'] = past_data_with_predictions['voter_predictions'].replace(-1, 0)
        correct_voter = sum(past_data_with_predictions['voter_predictions'] == past_data_with_predictions['WL_encoded'])
        total_voter = len(past_data_with_predictions['voter_predictions'].dropna())
        accuracy_voter = round((correct_voter / total_voter * 100) if total_voter != 0 else 0, 2)
        
        # Display accuracies
        st.write(f"LSTM Prediction Accuracy: {accuracy_lstm}%")
        st.write(f"LSTM (5 Game Memory) Prediction Accuracy: {accuracy_lstm_seq}% **Started on 2023-11-01")
        st.write(f"Voter Prediction Accuracy: {accuracy_voter}%")
        
        # Display predictions and results
        st.write(past_data_with_predictions)
        
if __name__ == "__main__":
    main()
