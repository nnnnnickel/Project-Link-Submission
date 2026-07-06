# Agentic Tarot Reader

**This repository is for submissions to the AI Agents: Intensive Vibe Coding Capstone Project.**

We built a Google ADK-compatible tarot reader with Gemini generator, GraphRAG parquet retrieval, JSON session memory, Skill.md guidance, and a local Web UI.

## Requirements

- Python 3.10+
- A Google AI Studio API key, or Vertex AI environment variables

## Project Structure

```text
submission/
  app/
    agent.py
    agent_refactored.py
    cli.py
    web_server.py
    graphrag_tool.py
    tarot_memory.py
    skill_registry.py
    card_input_normalizer.py
    static/
    skills/
  graphrag/
    entities.parquet
    relationships.parquet
    community_reports.parquet
    ...
  .env.example
  requirements.txt
  agents-cli-manifest.yaml
```

## Setup

Run these commands from the `submission` directory.

```powershell
pip install -r requirements.txt
```

Edit `.env` and replace `YOUR API KEY` with your Google API key.


## Run CLI

```bash
python -m app.cli --query "How is my love life today? My sign is Gemini." --cards "The Moon, The Empress reversed, Wheel of Fortune" --reset-session
```

## Run Web UI

```bash
python -m app.web_server
```

Then open: http://127.0.0.1:7860

The Web UI supports first readings, follow-up questions, and exiting the current session.
