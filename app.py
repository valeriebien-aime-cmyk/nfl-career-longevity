import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="NFL Career Longevity Predictor",
    page_icon="🏈",
    layout="wide",
)

@st.cache_resource
def load_artifacts():
    model_full = joblib.load('best_model_full.pkl')
    model_skill = joblib.load('best_model_skill.pkl')
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model_full, model_skill, metadata

model_full, model_skill, metadata = load_artifacts()

POSITION_NAMES = {
    'QB': 'Quarterback (QB)',
    'RB': 'Running Back (RB)',
    'WR': 'Wide Receiver (WR)',
    'TE': 'Tight End (TE)',
    'T': 'Offensive Tackle (T)',
    'G': 'Offensive Guard (G)',
    'C': 'Center (C)',
    'OL': 'Offensive Lineman (OL)',
    'FB': 'Fullback (FB)',
    'DE': 'Defensive End (DE)',
    'DT': 'Defensive Tackle (DT)',
    'NT': 'Nose Tackle (NT)',
    'DL': 'Defensive Lineman (DL)',
    'LB': 'Linebacker (LB)',
    'ILB': 'Inside Linebacker (ILB)',
    'OLB': 'Outside Linebacker (OLB)',
    'CB': 'Cornerback (CB)',
    'S': 'Safety (S)',
    'DB': 'Defensive Back (DB)',
    'K': 'Kicker (K)',
    'P': 'Punter (P)',
    'LS': 'Long Snapper (LS)',
}

position_options = sorted(metadata['positions'])
position_labels = [POSITION_NAMES.get(p, p) for p in position_options]



st.title("🏈 NFL Career Longevity Predictor")
st.markdown("**Predicting how long a drafted player will last in the NFL using draft information and rookie-year performance**")
st.markdown("---")

st.sidebar.header("Player Profile")
st.sidebar.markdown("Enter draft information and rookie year stats:")

st.sidebar.subheader("Draft Information")
qb_idx = position_options.index('QB') if 'QB' in position_options else 0
selected_label = st.sidebar.selectbox("Position", position_labels, index=qb_idx)
position = position_options[position_labels.index(selected_label)]
draft_round = st.sidebar.slider("Draft Round", 1, 7, 1)
pick = st.sidebar.slider("Overall Pick Number", 1, 260, 15)
age = st.sidebar.slider("Age at Draft", 20, 30, 22)

st.sidebar.subheader("Rookie Year Stats")
games_played = st.sidebar.slider("Games Played", 0, 17, 12)

is_skill = position in metadata['skill_positions']

if position == 'QB':
    pass_attempts = st.sidebar.number_input("Pass Attempts", 0, 700, 350)
    pass_yards = st.sidebar.number_input("Pass Yards", 0, 6000, 2500)
    pass_tds = st.sidebar.number_input("Pass TDs", 0, 60, 15)
    interceptions = st.sidebar.number_input("Interceptions", 0, 30, 10)
    carries, rush_yards, rush_tds = 0, 0, 0
    targets, receptions, rec_yards, rec_tds = 0, 0, 0, 0
elif position == 'RB':
    pass_attempts, pass_yards, pass_tds, interceptions = 0, 0, 0, 0
    carries = st.sidebar.number_input("Carries", 0, 400, 150)
    rush_yards = st.sidebar.number_input("Rush Yards", 0, 2500, 600)
    rush_tds = st.sidebar.number_input("Rush TDs", 0, 25, 5)
    targets = st.sidebar.number_input("Targets", 0, 150, 30)
    receptions = st.sidebar.number_input("Receptions", 0, 120, 22)
    rec_yards = st.sidebar.number_input("Receiving Yards", 0, 1500, 180)
    rec_tds = st.sidebar.number_input("Receiving TDs", 0, 15, 1)
elif position in ['WR', 'TE']:
    pass_attempts, pass_yards, pass_tds, interceptions = 0, 0, 0, 0
    carries, rush_yards, rush_tds = 0, 0, 0
    targets = st.sidebar.number_input("Targets", 0, 200, 80)
    receptions = st.sidebar.number_input("Receptions", 0, 150, 50)
    rec_yards = st.sidebar.number_input("Receiving Yards", 0, 2000, 600)
    rec_tds = st.sidebar.number_input("Receiving TDs", 0, 20, 4)
else:
    pass_attempts, pass_yards, pass_tds, interceptions = 0, 0, 0, 0
    carries, rush_yards, rush_tds = 0, 0, 0
    targets, receptions, rec_yards, rec_tds = 0, 0, 0, 0
    st.sidebar.info(f"For {position}, only draft features and games played are used.")

ypa = pass_yards / pass_attempts if pass_attempts >= 50 else 0
ypc = rush_yards / carries if carries >= 20 else 0
catch_rate = receptions / targets if targets >= 20 else 0
ypt = rec_yards / targets if targets >= 20 else 0
total_tds = pass_tds + rush_tds + rec_tds

position_encoded_map = {p: i for i, p in enumerate(sorted(metadata['positions']))}
position_encoded = position_encoded_map.get(position, 0)

feature_row = pd.DataFrame([{
    'round': draft_round, 'pick': pick, 'age': age, 'position_encoded': position_encoded,
    'games_played': games_played, 'pass_attempts': pass_attempts, 'pass_yards': pass_yards,
    'pass_tds': pass_tds, 'interceptions': interceptions, 'carries': carries,
    'rush_yards': rush_yards, 'rush_tds': rush_tds, 'targets': targets,
    'receptions': receptions, 'rec_yards': rec_yards, 'rec_tds': rec_tds,
    'total_tds': total_tds, 'yards_per_attempt': ypa, 'yards_per_carry': ypc,
    'catch_rate': catch_rate, 'yards_per_target': ypt,
}])[metadata['combined_features']]

chosen_model = model_skill if is_skill else model_full
model_label = "Skill-Position Model" if is_skill else "Full Model"

prediction = chosen_model.predict(feature_row)[0]
probabilities = chosen_model.predict_proba(feature_row)[0]
classes = chosen_model.classes_

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    color_map = {'Short': '#e74c3c', 'Medium': '#3498db', 'Long': '#2ecc71', 'Elite': '#f39c12'}
    pred_color = color_map.get(prediction, '#FFFFFF')
    
    st.markdown(f"""
    <div style="background-color: #F0F4F8; padding: 30px; border-radius: 8px; border-left: 6px solid {pred_color};">
        <p style="color: #5A6C82 !important; margin-bottom: 8px; font-size: 13px; font-weight: 600;">PREDICTED CAREER LENGTH</p>
        <h1 style="color: {pred_color} !important; margin: 0; font-size: 64px; font-weight: 700;">{prediction}</h1>
        <p style="color: #1A2332 !important; margin-top: 8px; font-size: 18px; font-weight: 500;">{
            '0–2 seasons' if prediction == 'Short' else
            '3–5 seasons' if prediction == 'Medium' else
            '6–8 seasons' if prediction == 'Long' else
            '9+ seasons'
        }</p>
        <p style="color: #5A6C82 !important; margin-top: 16px; font-size: 13px;">Using: {model_label}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("Probability Breakdown")
    
    prob_df = pd.DataFrame({'Class': classes, 'Probability': probabilities})
    prob_df = prob_df.sort_values('Probability', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=prob_df['Probability'],
        y=prob_df['Class'],
        orientation='h',
        marker=dict(color=[color_map.get(c, '#FFFFFF') for c in prob_df['Class']]),
        text=[f"{p:.1%}" for p in prob_df['Probability']],
        textposition='outside',
        textfont=dict(color='#1A2332', size=14),
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1A2332', size=14),
        xaxis=dict(range=[0, 1], tickformat='.0%', gridcolor='#E8EEF5', color='#1A2332'),
        yaxis=dict(gridcolor='#E8EEF5', color='#1A2332'),
        margin=dict(l=10, r=40, t=20, b=20),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("Model Performance")
perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.markdown("""
    <div class="metric-card">
        <p style="color: #5A6C82 !important; margin: 0; font-size: 12px; font-weight: 600;">FULL MODEL · MACRO F1</p>
        <h2 style="color: #2ecc71 !important; margin: 8px 0; font-size: 42px;">0.359</h2>
        <p style="color: #1A2332 !important; margin: 0; font-size: 13px;">Combined + Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

with perf_col2:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #3498db;">
        <p style="color: #5A6C82 !important; margin: 0; font-size: 12px; font-weight: 600;">SKILL POSITIONS · MACRO F1</p>
        <h2 style="color: #3498db !important; margin: 8px 0; font-size: 42px;">0.405</h2>
        <p style="color: #1A2332 !important; margin: 0; font-size: 13px;">Combined + Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

with perf_col3:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #f39c12;">
        <p style="color: #5A6C82 !important; margin: 0; font-size: 12px; font-weight: 600;">RANDOM BASELINE</p>
        <h2 style="color: #f39c12 !important; margin: 8px 0; font-size: 42px;">0.250</h2>
        <p style="color: #1A2332 !important; margin: 0; font-size: 13px;">4-class random guessing</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Feature Set Comparison")

comparison_data = pd.DataFrame({
    'Feature Set': ['Draft-Only', 'Rookie-Only', 'Combined'] * 2,
    'Group': ['Full Dataset'] * 3 + ['Skill Positions Only'] * 3,
    'Macro F1': [0.328, 0.225, 0.359, 0.324, 0.388, 0.405],
})

fig2 = go.Figure()
for group, color in [('Full Dataset', '#2ecc71'), ('Skill Positions Only', '#3498db')]:
    sub = comparison_data[comparison_data['Group'] == group]
    fig2.add_trace(go.Bar(
        name=group,
        x=sub['Feature Set'],
        y=sub['Macro F1'],
        marker_color=color,
        text=[f"{v:.3f}" for v in sub['Macro F1']],
        textposition='outside',
        textfont=dict(color='#1A2332', size=14),
    ))

fig2.update_layout(
    barmode='group',
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#1A2332', size=14),
    xaxis=dict(gridcolor='#E8EEF5', color='#1A2332'),
    yaxis=dict(range=[0, 0.5], gridcolor='#E8EEF5', color='#1A2332', title=dict(text='Macro F1 Score', font=dict(color='#1A2332'))),
    legend=dict(bgcolor='#F0F4F8', font=dict(color='#1A2332')),
    height=400,
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("**DATA 6545 Final Project · Spring 2026 · Valerie Bien-Aime**")
