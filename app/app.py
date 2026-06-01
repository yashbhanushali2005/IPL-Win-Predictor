import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import base64
import streamlit.components.v1 as components

def speak(text):
    speech_html = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.rate = 1;
        msg.pitch = 1;
        msg.volume = 1;
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(speech_html, height=0)

def play_sound(file_path):
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()

        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Fails silently if audio file is missing

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="IPL Live Predictor",
    page_icon="🏏",
    layout="wide"
)

# ---------------- BACKGROUND IMAGE FIX ---------------- #
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

bg_image = get_base64_image("app/assets/bg.jpg")

if bg_image:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Glass panel */
    .block-container {{
        background: rgba(0,0,0,0.70);
        padding: 2rem;
        border-radius: 15px;
    }}
    
    /* Header */
    h1 {{
        text-align: center;
        color: #00ff99 !important;
        font-size: 40px;
    }}
    
    h3 {{
        text-align: center;
        color: white !important;
    }}
    
    /* Buttons */
    .stButton button {{
        background-color: #00cc66;
        color: white;
        border-radius: 10px;
        height: 3em;
    }}
    
    .stButton button:hover {{
        background-color: #00ff99;
        color: black;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open("ipl_win_model.pkl", "rb"))

# ---------------- HEADER (CRICBUZZ STYLE) ---------------- #
st.markdown("# 🏏 IPL LIVE WIN PREDICTOR")
st.markdown("### 🔴 Cricbuzz-style Live Match Analytics Dashboard")

st.markdown("---")

# ---------------- TEAMS ---------------- #
teams = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
    "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals",
    "Punjab Kings", "Sunrisers Hyderabad",
    "Gujarat Titans", "Lucknow Super Giants"
]

colA, colB = st.columns(2)

with colA:
    batting_team = st.selectbox("🏏 Batting Team", teams)

with colB:
    bowling_team = st.selectbox("🎯 Bowling Team", [t for t in teams if t != batting_team])

st.markdown("---")

# ---------------- SCOREBOARD STRIP ---------------- #
st.markdown(f"### 🟢 LIVE: {batting_team} vs {bowling_team}")
st.markdown("---")

# ---------------- INPUTS ---------------- #
col1, col2, col3, col4 = st.columns(4)

with col1:
    runs_left = st.number_input("Runs Left", min_value=0, value=50)

with col2:
    balls_left = st.number_input("Balls Left", min_value=0, value=30)

with col3:
    wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10, value=5)

with col4:
    current_score = st.number_input("Current Score", min_value=0, value=120)

st.markdown("---")

# ---------------- FEATURES ---------------- #
if balls_left > 0:
    current_run_rate = (current_score * 6) / (120 - balls_left) if (120 - balls_left) > 0 else 0
    required_run_rate = (runs_left * 6) / balls_left
else:
    current_run_rate = 0
    required_run_rate = 0

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Get Live Win Probability"):

    play_sound("app/assets/crowd.mp3")

    # ---------------- HARD RULES ---------------- #
    if runs_left == 0:
        st.success(f"🏆 {batting_team} has already chased the target!")
        st.balloons()
        st.stop()

    if wickets_left == 0 or balls_left == 0:
        st.error(f"❌ {bowling_team} wins the match!")
        st.stop()

    # ---------------- MODEL PREDICTION ---------------- #
    input_data = np.array([[runs_left, balls_left, wickets_left, current_run_rate, required_run_rate]])
    result = model.predict_proba(input_data)

    win_prob = result[0][1]
    loss_prob = result[0][0]

    win_percent = round(win_prob * 100, 2)
    loss_percent = round(loss_prob * 100, 2)

    # ---------------- LIVE SCOREBOARD ---------------- #
    st.markdown("## 📊 Live Win Meter")
    st.progress(int(win_percent))

    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric(f"🏏 {batting_team}", f"{win_percent}%")
    with col_metric2:
        st.metric(f"🎯 {bowling_team}", f"{loss_percent}%")

    # ---------------- PIE CHART ---------------- #
    fig = go.Figure(data=[
        go.Pie(
            labels=[batting_team, bowling_team],
            values=[win_percent, loss_percent],
            hole=0.6,
            marker_colors=["#00ff99", "#ff4b4b"]
        )
    ])

    fig.update_layout(
        title="📊 Win Probability Breakdown",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- COMMENTARY STYLE INSIGHT ---------------- #
    st.markdown("---")
    st.subheader("🧠 Cricbuzz Live Commentary")

    if win_percent > 75:
        commentary = f"🔥 {batting_team} are absolutely cruising towards victory with {win_percent} percent win probability!"
        st.success(commentary)
    elif win_percent > 50:
        commentary = f"⚔️ Match is slightly in favor of {batting_team}, but {bowling_team} is still fighting back strongly!"
        st.warning(commentary)
    else:
        commentary = f"💥 {bowling_team} is currently dominating the game. {batting_team} needs something special to come back!"
        st.error(commentary)
        
    # Trigger Text-to-Speech
    speak(commentary)

    # ---------------- WINNER PREDICTION ---------------- #
    predicted_winner = batting_team if win_percent > 50 else bowling_team
    st.info(f"🏆 Predicted Winner: {predicted_winner}")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("🏏 Cricbuzz-style UI | Machine Learning Model | IPL Dataset")