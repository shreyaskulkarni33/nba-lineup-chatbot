# NBA Lineup Analytics Assistant

## Overview
I built an interactive NBA lineup analytics assistant that answers natural-language questions using a cleaned dataset of NBA 5-man lineup performance from the 2023–2024 season. The app organizes lineup-level metrics such as minutes played, pace, and efficiency ratings into analysis-ready summaries, allowing users to explore how lineup stability and style of play relate to performance.

Rather than generating free-form answers, the assistant is grounded in computed statistics that are explicitly provided to the model, keeping responses transparent, reproducible, and data-backed.

## What you can ask
Examples of supported questions:
- Do high-minute lineups tend to perform better?
- How does lineup performance change across minute buckets?
- Does pace relate to offensive efficiency?
- Which lineups perform best among those with significant playing time?

## How it works
- The dataset is cleaned and standardized using pandas.
- Lineups are grouped into minute buckets to reflect lineup stability and coach trust.
- Summary tables are computed for performance and pace.
- A language model translates these summaries into natural-language explanations.
- The AI is intentionally scoped to interpretation, not computation.

## Tech stack
- Python
- pandas / numpy
- Streamlit
- OpenAI API

## Live demo
👉 **[https://nba-lineup-insights.streamlit.app]**
