---
title: MariMAMBA
emoji: 🍄
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "6.12.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🍄 MariMAMBA — Physics-Guided AI Level Generator & Fixer

MariMAMBA is a conditional Mamba state-space model trained to generate Super Mario Bros levels from high-level attributes (enemies, gaps, pipes). Generated levels are then validated using a BFS physics engine and repaired by an LLM-powered fix pipeline.

## Features
- **Conditional Generation:** Control the number of enemies, gaps, and pipes.
- **Physics-Guided BFS Solver:** Automatically checks if Mario can traverse the generated level.
- **AI Fix Engine:** Uses a Groq-hosted LLM to repair unsolvable levels with minimal tile edits.
- **Live Visualization:** See the BFS solution path, stuck points, and fix rounds in real time.

## Usage
1. Set the generation parameters (enemies, gaps, pipes, temperature, etc.).
2. Optionally provide a **Groq API Key** to enable the AI fix engine.
3. Click **GENERATE + FIX** and watch the level get generated and solved live.
