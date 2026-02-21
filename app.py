
import json
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI


#Here, we will leverage the streamlit package to set up a title, layout, caption and add an emoji as well. 

st.set_page_config(page_title="NBA Lineup Chatbot", layout="wide")
st.title("🏀 NBA Lineup Analytics Chatbot")
st.markdown(
    """
**Project overview**

I built an interactive NBA lineup analytics chatbot that answers natural-language questions using a cleaned dataset of NBA 5-man lineup performance. The app loads lineup-level features such as minutes played, pace, and efficiency ratings and organizes them into analysis-ready summaries, allowing users to explore how lineup performance changes with playing time or pace. Rather than generating free-form answers, the chatbot is grounded in computed statistics that I explicitly provide, keeping responses transparent, reproducible, and data-backed. I handled the data cleaning, feature selection, aggregation logic, and overall product and UX design, while the AI layer is intentionally scoped to translating these precomputed summaries into clear, natural-language explanations.

"""
)

st.markdown("Analyze NBA 5-man lineups from the 2023–2024 season using minutes played, pace, and efficiency metrics, grouped into minute buckets to study lineup stability and performance. Future iterations will add team, opponent, and positional context.")
           



#I've purchased an OpenAPI key to leverage for this chatbot, however it will be stored in a secret file. I've also included a prompt and included Trust and Safety guardrails so questions stay pertinent to NBA Analytics  

 

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM = """
You are an NBA lineup analytics assistant.

Data dictionary:
- group_name: the 5-player lineup (player names)
- min: total minutes played by that lineup
- net_rating: points per 100 possessions (offense - defense) for that lineup
- off_rating: points scored per 100 possessions
- def_rating: points allowed per 100 possessions
- pace: estimated possessions per 48 minutes
- minute_bucket: a categorical bucket based on min (e.g., 0–50, 50–100, ...)

Use only the provided context tables to answer.
If the context does not include the needed data, say what table/summary is missing.
Be concise and cite numbers from the tables.
"""


#Here, we are loading the data. The CSV file was created by - loading the raw data from the NBA API, cleaning and standardizing columns, and adding feature engineering including minute buckets to answer questions like "Are high minute lineups more efficient than low minute lineups? The csv below is a final product of all that work. 


@st.cache_data
def load_data():
    df = pd.read_csv("nba_5man_lineups_features.csv")

    for c in ["min", "net_rating", "off_rating", "def_rating", "pace"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "minute_bucket" not in df.columns:
        bins = [0, 50, 100, 200, 400, np.inf]
        labels = ["0–50", "50–100", "100–200", "200–400", "400+"]
        df["minute_bucket"] = pd.cut(df["min"], bins=bins, labels=labels, right=False)

    return df

df = load_data()


#Helper functions overall_summary provides a dataset level snapshot to give quick context about the scale. bucket_summary groups the lineups into predefined minutes brackets. The pace one  Summarizes pace vs offensive rating by minute_bucket. This gives the model real numbers to answer pace-related questions.The top lineups method shows the top lineups by net_rating, including group_name (players), minutes, pace, and ratings. This helps the model answer player/lineup questions.

def overall_summary():
    return {
        "total_lineups": int(len(df)),
        "avg_net_rating": round(df["net_rating"].mean(), 2),
        "avg_minutes": round(df["min"].mean(), 1),
    }

def bucket_summary():
    out = (
        df.groupby("minute_bucket")
          .agg(
              lineups=("net_rating", "count"),
              avg_net_rating=("net_rating", "mean"),
              avg_minutes=("min", "mean"),
          )
          .reset_index()
    )
    out["avg_net_rating"] = out["avg_net_rating"].round(2)
    out["avg_minutes"] = out["avg_minutes"].round(1)
    return out

def pace_offense_summary():
    out = (
        df.dropna(subset=["minute_bucket", "pace", "off_rating"])
          .groupby("minute_bucket")
          .agg(
              lineups=("off_rating", "count"),
              avg_pace=("pace", "mean"),
              avg_off_rating=("off_rating", "mean"),
          )
          .reset_index()
    )
    out["avg_pace"] = out["avg_pace"].round(2)
    out["avg_off_rating"] = out["avg_off_rating"].round(2)
    return out

def top_lineups_preview(n=15, min_minutes=100):
    cols = [c for c in ["group_name", "min", "pace", "net_rating", "off_rating", "def_rating", "minute_bucket"] if c in df.columns]
    out = (
        df[df["min"] >= min_minutes]
        .dropna(subset=["net_rating"])
        .sort_values("net_rating", ascending=False)
        .loc[:, cols]
        .head(n)
    )
    # round for readability
    for c in ["min", "pace", "net_rating", "off_rating", "def_rating"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out

#The ask function will connect the users natural language question to the NBA lineup data. I didn't want the language model to answer freely so I leveraged AI here to help me inject computed summaries from the dataset into the prompt. 


def ask(question):
    context = f"""
Overall summary:
{overall_summary()}

By minute bucket:
{bucket_summary().to_string(index=False)}


Pace vs offensive rating by minute bucket:
{pace_offense_summary().to_string(index=False)}

Top lineups (includes group_name = players):
{top_lineups_preview(n=15, min_minutes=100).to_string(index=False)}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "user", "content": context},
        ],
    )

    return response.output_text


#This section defines the interactive user interface of the app, providing a text input area, a button to submit and results. 

question = st.text_area(
    "Ask a question",
    "Do high-minute lineups tend to be elite?",
    height=100,
)

if st.button("Ask"):
    with st.spinner("Thinking..."):
        answer = ask(question)
    st.subheader("Answer")
    st.write(answer)

st.divider()

st.subheader("Sample of lineup data")
st.caption("First 5 rows of the NBA 5-man lineup dataset")
st.dataframe(df.head(), width="stretch")





