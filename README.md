# 🔎 AI Research Agent (CrewAI + Groq + Streamlit)

A beginner-friendly single-agent AI app: you type a topic, a CrewAI agent
searches the web with **free** DuckDuckGo search, and writes a structured
report using Groq's `openai/gpt-oss-120b` model — all in a Streamlit UI.

This version reads the Groq API key **only from Streamlit secrets**, so
it's built to go straight from GitHub to Streamlit Community Cloud.

## Project structure

```
research_agent/
├── app.py                          # Streamlit UI (entry point)
├── agent.py                        # CrewAI agent, task, and crew setup
├── tools/
│   ├── __init__.py
│   └── duckduckgo_tool.py          # Free DuckDuckGo search tool (no API key)
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── secrets.toml.example        # Template only — never commit real secrets.toml
```

## Step 1 — Get a free Groq API key

1. Go to https://console.groq.com/keys
2. Sign up / log in, click **"Create API Key"**, and copy it. Keep it
   somewhere safe — you'll paste it into Streamlit in Step 3, not into
   any file in this repo.

## Step 2 — Push this project to GitHub

1. Create a new, empty repository on GitHub (don't add a README there —
   you already have one).
2. From inside this project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AI research agent"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
3. `.gitignore` already excludes `.streamlit/secrets.toml`, so if you
   ever create a real one locally, it won't get pushed by accident.

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **"Create app"** → **"From existing repo"**.
3. Pick your repo, branch `main`, and set the main file path to `app.py`.
4. Before clicking Deploy (or right after, via the app's **⋮ menu →
   Settings → Secrets**), paste in:
   ```toml
   GROQ_API_KEY = "your_real_groq_api_key"
   ```
5. Click **Save**, then **Deploy**. Streamlit Cloud installs
   `requirements.txt` automatically and starts the app.

That's it — the app reads `GROQ_API_KEY` from `st.secrets` at runtime
(see `get_groq_api_key()` in `app.py`). If the key is missing, the app
shows a clear warning instead of crashing, and disables the button.

## How it works (short version)

- `tools/duckduckgo_tool.py` wraps the free `ddgs` package (the current
  name for what used to be called `duckduckgo-search`) as a CrewAI tool.
  No API key is needed for search.
- `agent.py` creates **one** CrewAI `Agent` ("Senior Research Analyst")
  with that search tool, a `Task` describing what a good report looks
  like, and a `Crew` that runs the agent (`Process.sequential`, since
  there's only one agent).
- The agent's `LLM` is `groq/openai/gpt-oss-120b` — CrewAI routes any
  non-native provider (Groq included) through LiteLLM automatically, so
  there's nothing extra to install for that.
- `app.py` is a small Streamlit form that calls `agent.run_research()`
  and displays / lets you download the resulting Markdown report.

## Updating your secret later

If your Groq key ever changes or expires: open your app on Streamlit
Cloud → **⋮ menu → Settings → Secrets**, edit the value, save. The app
restarts automatically and picks up the new key — no code change or
redeploy needed.

## Troubleshooting

- **App shows "No GROQ_API_KEY found in Streamlit secrets"**: you
  deployed before adding the secret, or there's a typo in the key name.
  Go to Settings → Secrets and confirm it's exactly `GROQ_API_KEY`.
- **"Search failed" messages in the report**: DuckDuckGo occasionally
  rate-limits or blocks requests from cloud IPs. Wait a bit and retry,
  or lower `max_results` in `tools/duckduckgo_tool.py`.
- **Groq errors about the model name**: double-check your key is valid
  and that `openai/gpt-oss-120b` is still listed at
  https://console.groq.com/docs/models — Groq occasionally renames or
  retires models.

## Next steps (optional, once you're comfortable)

- Add more tools (e.g. a Wikipedia tool) to `researcher.tools`.
- Split the single task into two tasks/agents (Researcher → Writer).
- Cache reports by topic with `st.cache_data` to avoid re-running the
  same query twice.
