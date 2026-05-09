# Multi-Agent Study Planner

This project is a self-contained Python demo of a LangGraph-style multi-agent workflow.
It takes a user request, extracts basic requirements, builds a plan, generates research notes,
writes a structured response, and runs a simple critique pass.

## Features

- Single-file implementation in `multi_agent_system.py`
- Optional `langgraph` support if installed
- Local fallback graph runner when optional dependencies are missing
- CLI input via prompt or `--request`
- JSON output via `--json`

## Requirements

- Python 3.10 or newer
- Optional: `langgraph`

## Run

From the project root:

```bash
python3 multi_agent_system.py
```

Enter a request when prompted, or pass it directly:

```bash
"help me create a project plan for a team in 2 weeks"
```

Print the final state as JSON:

```bash
python3 multi_agent_system.py --request "create a study plan for Python" --json
```

## Example Requests

- `Help me create a study plan for learning Python in 2 weeks.`
- `Give me a project plan for a small team building a mobile app.`
- `Make a resume review checklist for a beginner applying for internships.`

## How It Works

The workflow runs in this order:

1. Parse the user request into basic fields such as topic, audience, skill level, and time available.
2. Extract goals and constraints.
3. Build a simple execution plan.
4. Generate research notes.
5. Write the final response.
6. Critique the output for completeness.

## Project Structure

```text
multi_agent_system.py
README.md
```

## Notes

- The "agents" are deterministic Python functions, not live LLM agents.
- The design is intentionally simple so it can run in restricted environments.
- If `langgraph` is installed, the script will use it; otherwise it falls back to a local graph runner.
