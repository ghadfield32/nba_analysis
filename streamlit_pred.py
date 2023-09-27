import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load the data
def load_data():
    data = pd.read_csv(r"C:\Users\ghadf\OneDrive\Desktop\Data Analytics\Python\ML\nba_w_l_prediction_models\nba_analysis\data\season_pred.csv")  # Modify the path to your CSV
    return data

# Load the data
data = load_data()

# App title and intro
st.title("NBA Team Analysis")
st.write("An analysis based on CSV data for NBA Team Predictions and previous season averages by home/away game.")

# Display a table with data
if st.checkbox("Show Raw Data"):
    st.write(data)

# Scatter plot
teams = data["TEAM_NAME"].unique()
team_choice = st.selectbox("Select a Team for Scatter Plot Analysis:", teams)

team_data = data[data["TEAM_NAME"] == team_choice]
fig, ax = plt.subplots()
ax.scatter(team_data["FG_PCT"], team_data["PLUS_MINUS"], label="Field Goal % vs. Plus Minus")
ax.set_xlabel("Field Goal %")
ax.set_ylabel("Plus Minus")
ax.set_title(f"Scatter Plot for {team_choice}")
st.pyplot(fig)

# Bar chart
if st.checkbox("Show Bar Chart:"):
    metrics = ["FG_PCT", "FG3_PCT", "FT_PCT", "TS%", "eFG%", "AST%"]
    metric_choice = st.selectbox("Select a metric:", metrics)
    fig, ax = plt.subplots()
    data.groupby("TEAM_NAME")[metric_choice].mean().sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel(metric_choice)
    ax.set_title(f"Average {metric_choice} by Team")
    st.pyplot(fig)

# Upcoming Predictions Table
st.subheader("Upcoming Predictions")
home_team = st.selectbox("Select Home Team:", [""] + list(teams))  # added an empty choice to force users to select a team
away_team = st.selectbox("Select Away Team:", [""] + list(teams))

if home_team and away_team:  # ensures both dropdowns have a selection
    prediction_data = data[(data["TEAM_NAME"] == home_team) & (data["TEAM_NAME_OPP"] == away_team)]  # assuming you have a "TEAM_NAME_OPP" column for the opponent team
    if not prediction_data.empty:
        st.write("Prediction for the matchup:")
        st.write(prediction_data[["TEAM_NAME", "TEAM_NAME_OPP", "PREDICTION"]])  # adjust columns as needed
    else:
        st.write(f"No available prediction for {home_team} vs. {away_team}.")
