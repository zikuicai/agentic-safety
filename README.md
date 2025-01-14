# agentic-safety


## Project Overview
A unified and scalable agentic workflow that addresses multifaceted safety concerns (sensitive/dangerous information leakage, jailbreakinng).


## Environment Setup
To run this project, follow these steps:

### Prerequisites
- Python 3.10 or higher
- Access to LLM APIs (e.g., OpenAI GPT-4o, Claude-3.5) or open-source pre-trained models
- DSPy framework 


### Install Required Libraries
Install the following Python libraries:

- conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
- pip install huggingface-hub
- datasets
- dotenv
- pip install hydra-core
- pip install git+https://github.com/stanfordnlp/dspy.git

### Required Datasets

unlearning: wmdp, mmlu, mt-bench, tofu, harry potter


### Directory Structure
- `configs/`: Contains configuration files for the project.

- `sim/`: Contains the main simulation logic.
    - `base.py`: Base Simulator (abstract) class logic.
    - `dspy_sim.py`: Script our current dspy based simulation logic.

- `utils/`: Contains utility functions used across the project.

- `evals/`: Contains benchmarks and their evaluation code.

- `.env`: Environment file to store API keys and other sensitive information. Make sure to create this file and add your
  API keys.

- `run.sh`: Contains commands to run code and reporduce results.
