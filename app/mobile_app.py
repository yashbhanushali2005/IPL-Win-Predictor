import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="IPL Mobile Predictor",
    page_icon="🏏",
    layout="centered"
)

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open("ipl_win_model.pkl", "rb"))

# ---------------- TITLE ---------------- #
st.title("🏏 IPL Mobile Win Predictor")
st.markdown("### 📱 Simple & Fast Match Prediction")

st.markdown("---")

# ---------------- TEAMS ---------------- #
teams = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
    "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals",
    "Punjab Kings", "Sunrisers Hyderabad",
    "Gujarat Titans", "Lucknow Super Giants"
]

batting_team = st.selectbox("🏏 Batting Team", teams)
bowling_team = st.selectbox("🎯 Bowling Team", [t for t in teams if t != batting_team])

st.markdown("---")

# ---------------- INPUTS (MOBILE STACKED) ---------------- #
runs_left = st.number_input("Runs Left", min_value=0)
balls_left = st.number_input("Balls Left", min_value=0)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10)
current_score = st.number_input("Current Score", min_value=0)

st.markdown("---")

# ---------------- FEATURES ---------------- #
if balls_left > 0:
    current_run_rate = (current_score * 6) / (120 - balls_left)
    required_run_rate = (runs_left * 6) / balls_left
else:
    current_run_rate = 0
    required_run_rate = 0

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict Match Result"):

    # ---------------- RULES ---------------- #
    if runs_left == 0:
        st.success(f"🏆 {batting_team} already won the match!")
        st.balloons()
        st.stop()

    if wickets_left == 0 or balls_left == 0:
        st.error(f"❌ {bowling_team} wins the match!")
        st.stop()

    # ---------------- MODEL ---------------- #
    input_data = np.array([[runs_left, balls_left, wickets_left,
                            current_run_rate, required_run_rate]])

    result = model.predict_proba(input_data)

    win_percent = round(result[0][1] * 100, 2)
    loss_percent = round(result[0][0] * 100, 2)

    # ---------------- RESULT ---------------- #
    st.markdown("## 📊 Match Result")

    st.metric(f"🏏 {batting_team}", f"{win_percent}%")
    st.metric(f"🎯 {bowling_team}", f"{loss_percent}%")

    st.progress(int(win_percent))

    # ---------------- PIE CHART ---------------- #
    fig = go.Figure(data=[
        go.Pie(
            labels=[batting_team, bowling_team],
            values=[win_percent, loss_percent],
            hole=0.5,
            marker_colors=["#00ff99", "#ff4b4b"]
        )
    ])

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- INSIGHT ---------------- #
    st.markdown("---")
    st.subheader("🧠 Match Insight")

    if win_percent > 70:
        st.success(f"🔥 {batting_team} is dominating!")
    elif win_percent > 50:
        st.warning(f"⚔️ Slight edge to {batting_team}")
    else:
        st.error(f"💥 {bowling_team} is controlling the match!")

    st.info(f"🏆 Predicted Winner: {batting_team if win_percent > 50 else bowling_team}")

    st.markdown("---")
    st.caption("📱 Mobile Optimized | IPL ML Predictor")