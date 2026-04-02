# CLARITY: Unmasking Political Question Evasions

[SemEval 2026 Challenge](https://konstantinosftw.github.io/CLARITY-SemEval-2026/#BibTeX) - Unmasking Political Question Evasions

This project focuses on identifying and classifying evasive responses in political interviews. We leverage state-of-the-art Large Language Models and advanced prompting techniques to provide a granular analysis of how politicians address (or avoid) direct questions.

## Dataset
We use the [QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion) dataset, which contains pairs of interview questions and answers from political figures.

- **Train Set**: 3448 examples
- **Test Set**: 308 examples

The responses are categorized into:
- **3 Macro-Categories (Clarity)**: *Ambivalent*, *Clear Reply*, *Clear Non-Reply*.
- **9 Evasion Techniques**: *Explicit*, *Dodging*, *Implicit*, *General*, *Deflection*, *Declining to answer*, *Claims ignorance*, *Clarification*, *Partial/half-answer*.

---

## Project Structure

### Task 0: Exploratory Data Analysis (EDA)
Comprehensive analysis of the QEvasion dataset, including:
- Statistics on the distribution of clarity and evasion labels.
- Text length analysis for both questions and answers.
- Correlation analysis between specific evasion techniques and macro-clarity categories.
- Reference: `Task0_Data_Analysis.ipynb`

### Task 1: Direct Clarity Classification
Fine-tuning of **Llama 3.1 8B-Instruct** to directly classify political responses into the three macro-categories.
- **Methodology**: 4-bit QLoRA + **DoRA** (Weight-Decomposed Low-Rank Adaptation).
- **Goal**: Direct mapping from text to clarity label.
- **Evaluation**: Classification reports and confusion matrices.
- Reference: `Task1.ipynb`

### Task 2: Evasion-Based Clarity Classification
A more granular approach where the model is fine-tuned to recognize the 9 specific evasion techniques.
- **Methodology**: Similar to Task 1, using QLoRA + DoRA on Llama 3.1 8B.
- **Strategy**: Predicting 9 classes and then mapping them deterministically to the clarity macro-categories.
- Reference: `Task2.ipynb`

### Task 2.1: Prompt Chaining & Question Decomposition
Advanced inference pipeline designed to overcome the limitations of standard classification when dealing with *multi-barrelled questions* (multiple questions in one turn).
1. **Splitter (Step 1)**: Decomposes a complex question into atomic sub-questions (sQA) using a structured JSON output.
2. **Evaluator (Step 2)**: Evaluates the politician's global answer against each sub-question individually.
3. **Aggregator (Step 3)**: A deterministic logic that detects partial responses. If at least one sub-question is answered explicitly while others are evaded, it assigns a `Partial/half-answer` label.
- Reference: `Task2_1.ipynb`

---

## Setup & Execution
Run the provided `setup.sh` to configure the environment and install dependencies listed in `requirements.txt`.

## Group Members
| Name      | Student ID | 
| :---        |    :----:   | 
| Stefano Zizzi     | s346595       | 
| Alessio Perrotti   | s346737        | 
| Riccardo Vaccari   | s348856        | 
| Davide Candela   | s347245        | 
| Luca Lodesani   | s346978      | 


