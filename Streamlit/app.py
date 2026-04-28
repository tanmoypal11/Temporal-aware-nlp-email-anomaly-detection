#================================Step 1: Load=======================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\Desktop\GUVI\Project\Development of a Temporal-Aware NLP Pipeline\email_anomaly.csv")
df['date'] = pd.to_datetime(df['date'])

df_anomalies = df[df['anomaly_label'] == -1]

st.title("📊 Email Anomaly Detection Dashboard")

#============================= SELECT BAR =========================================
option = st.selectbox(
    "Select Analysis",
    [
        "📈 Timeline",
        "⚠️ Top Risky Users",
        "🔑 Keyword Trend",
        "🚨 Flagged Emails"
    ]
)

#============================= 1. Timeline =========================================
if option == "📈 Timeline":
    st.subheader("Email Activity vs Anomalies")

    email_trend = df.groupby(df['date'].dt.date).size()
    anomaly_trend = df_anomalies.groupby(df_anomalies['date'].dt.date).size()

    fig, ax = plt.subplots()
    email_trend.plot(ax=ax, label="Total Emails")
    anomaly_trend.plot(ax=ax, label="Anomalies")
    ax.legend()

    st.pyplot(fig)

#============================= 2. Top Risky Users ==================================
elif option == "⚠️ Top Risky Users":
    st.subheader("Top Risky Users")

    risky_users = df_anomalies['from'].value_counts().head(10)
    st.bar_chart(risky_users)

#============================= 3. Keyword Trend ====================================
elif option == "🔑 Keyword Trend":
    st.subheader("Keyword Trend")

    keyword_trend = df.groupby(df['date'].dt.date)['keyword_flag'].sum()
    st.line_chart(keyword_trend)

#============================= 4. Flagged Emails ===================================
elif option == "🚨 Flagged Emails":
    st.subheader("Flagged Emails")

    st.dataframe(
        df_anomalies.sort_values(by='anomaly_score').head(50)
    )