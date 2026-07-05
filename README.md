# Agentic Tarot Reader

Google ADK-compatible tarot reader with Gemini generation, GraphRAG parquet retrieval, JSON session memory, optional Skill.md guidance, and a local Web UI.

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

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
New-Item -ItemType Directory -Force ..\tarot_project\graphrag-project-new\output | Out-Null
Copy-Item .\graphrag\* ..\tarot_project\graphrag-project-new\output -Recurse -Force
```

Edit `.env` and replace `YOUR API KEY` with your Google API key.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
mkdir -p ../tarot_project/graphrag-project-new/output
cp -R graphrag/. ../tarot_project/graphrag-project-new/output/
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

Then open:

```text
http://127.0.0.1:7860
```

The Web UI supports first readings, follow-up questions, and exiting the current session.

## Gemini Configuration

Use an AI Studio key in `.env`:

```text
GOOGLE_API_KEY=your_ai_studio_key
```

Or configure Vertex AI:

```text
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your_project
GOOGLE_CLOUD_LOCATION=global
```
