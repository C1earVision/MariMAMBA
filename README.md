# 🍄 MariMAMBA: Multi-Attribute Conditional Mamba for Level Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A high-precision level generation system using Selective State Space Models (Mamba) with multi-attribute controllability.**

MariMAMBA is a deep learning framework designed to generate playable, high-quality Super Mario Bros levels. By leveraging the **Mamba (SSM)** architecture with **FiLM (Feature-wise Linear Modulation)**, the system achieves fine-grained control over specific level features like enemy density, gap frequency, and pipe counts, outperforming transformer-based baselines in instruction adherence.

---

## ✨ Key Features

*   **⚡ Mamba Selective SSM**: Uses state-of-the-art Linear-Time Sequence Modeling for autoregressive column generation.
*   **🎯 Multi-Attribute Control**: Precise control via a 3-dimensional attribute vector `[Enemies, Gaps, Pipes]`.
*   **🧩 FiLM Modulation**: Hidden states are directly modulated by attribute embeddings at every layer for superior controllability.
*   **🛠️ Physics-Guided AI Fixer**: A robust refinement pipeline that detects "stuck points" using a BFS physics engine and repairs them via LLM-based structural repair (Groq API).
*   **📈 Auxiliary Loss Training**: Uses an `AttributePredictor` head and L1 loss to force the model to maintain feature awareness throughout the sequence.
*   **🔍 Detailed Evaluation**: Compare performance against baselines like **MarioGPT** using Mean Absolute Error (MAE) metrics.

---

## 🏗️ Architecture Overview

### 1. The Generative Backbone (`Mamba`)
The core model is a sequence-to-sequence Mamba architecture. Each column is processed through selective SSM layers. Unlike standard concatenation, we use **FiLM blocks** to scale and shift the Mamba hidden states based on the target attribute embedding, ensuring the instruction signal persists through deep layers.

### 2. Multi-Attribute Feature Vector
The model is conditioned on specific counts rather than a single difficulty scalar:
- **Enemies**: Total number of active enemies.
- **Gaps**: Number of bottomless pits.
- **Pipes**: Number of static pipe structures.

### 3. Training & Stability
- **Auxiliary Head**: An internal predictor head estimates attributes from hidden states during training.
- **L1 Loss Scaling**: Attribute errors are scaled to match Cross-Entropy magnitude, preventing gradient explosions.
- **CFG (Classifier-Free Guidance)**: Supports a "Null" token (`-1.0`) during training to allow for zero-instruction baseline comparison.

---

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/conditional-mamba.git
   cd conditional-mamba
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

---

## 🎮 Usage

### Training
To train the model with the multi-attribute auxiliary loss:
```bash
python -m scripts.train_mamba
```

### Generation (CLI)
Generate levels by specifying target counts for `[Enemies, Gaps, Pipes]`:
```bash
# Generate a level with 5 enemies, 0 gaps, and 2 pipes
python -m scripts.generate_levels --attributes "5,0,2" --columns 100 --cfg_scale 2.0
```

### Evaluation
Run the comparison script to see how the model stacks up against **MarioGPT**:
```bash
python -m scripts.compare_with_baseline
```
Run the formal evaluation to generate accuracy visualizations:
```bash
python -m scripts.evaluate_model
```

---

## 📂 Project Structure

```text
.
├── config/              # Training, Generation, and Eval YAMLs
├── data/                # LevelParser and Dataset loaders
├── dataset/             # Raw .txt Mario levels
├── evaluation/          # A* Agent and attribute judge logic
├── generation/          # Inference loop and PNG rendering
├── map_fixer_system/    # BFS engine and LLM repair pipeline
├── models/              # Mamba + FiLM architecture
├── scripts/             # CLI tools for training and evaluation
└── training/            # Trainer classes and loss scaling
```

---

## 🎨 Mario Level Studio (Gradio UI)

![Mario Level Studio UI](mario_level_studio_ui_1777849468001.png)

The project includes **Mario Level Studio**, a premium interactive interface built with Gradio. It allows you to:
- **Direct Control**: Precisely set target counts for Enemies, Gaps, and Pipes.
- **Visual Verification**: See the generated level and its solución path immediately.
- **AI-Powered Repair**: If a level is unsolvable, the **Physics-Guided Fixer** uses LLMs to structurally repair the level while you watch.
- **Debug View**: Visualize Mario's reachability via the BFS "heat map" overlay.

To launch the studio:
```bash
python demo.py
```

---

## 📊 Results (vs. MarioGPT)

| Metric | MarioGPT (Text) | MariMAMBA (Guided) |
| :--- | :--- | :--- |
| **Attribute Accuracy (MAE)** | ~1.25 | **~0.11** |
| **Playability Rate** | 100% | 100% |
| **Inference Speed** | Slow (Transformer) | **Fast (SSM)** |

MariMAMBA achieves nearly **10x higher accuracy** in following specific feature counts compared to text-based GPT baselines while maintaining perfect playability.

---

## Contributors
https://github.com/moaazaldakar --> Created The Map Fixer System

## License
MIT License

Copyright (c) 2026 Ahmed Ebrahim (C1earVision)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
