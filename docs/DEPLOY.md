# Deploying a public demo on Render (free plan)

The free plan has about 512 MB RAM and sleeps after ~15 minutes without traffic (the first request afterwards can take
up to a minute; the UI shows a waiting message). AMAN is built for this: no local ML models, prebuilt corpus,
embeddings only for the user's question.

## 1. Prepare the repository
- Commit `data/processed/chunks.jsonl` (the prebuilt corpus). Do not commit `data/raw/` or any key.
- If you have a vector index (`vectors_*.npy`), commit it too; without it AMAN runs keyword-only, which is fine for a first release.
- Check no secret is committed: `grep -rIn -E "AIza|AQ\.[A-Za-z0-9_-]{20,}" . --exclude-dir=.venv --exclude-dir=.git` must print nothing.

## 2. Create the service
1. Render → New → Blueprint → select the GitHub repo (it reads `render.yaml`).
2. When asked, set `GEMINI_API_KEY` (and optionally `AMAN_AUTHOR`, `AMAN_LINKEDIN`, `AMAN_GITHUB` for the About page).
3. Deploy, then open `/api/health` and try the chat.

## 3. Before sharing the link
- Put a spending/usage limit on the API key in Google AI Studio.
- Try: an Arabic question, an English question, a follow-up, a greeting, an off-topic question, a request to build malware (must be refused), thumbs up/down.
- Settings that protect the demo: `AMAN_RATE_PER_MIN` (per IP), `AMAN_DAILY_LLM_CAP` (then extractive answers).

## 4. Adding the vector index later
Free embedding quotas are small (about 1,000 passages per day at the time of writing), so run
`AMAN_EMBEDDER=gemini python -m aman.build_index` once a day; progress is saved and resumed. When it finishes,
commit the `vectors_*.npy` file and redeploy.
