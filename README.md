# AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security

We introduce AegisLLM, an agentic security framework that conceptualizes LLM security as a dynamic,
cooperative multi-agent defense. A structured society of autonomous agents—orchestrator, deflector,
responder, and evaluator—each performs specialized functions and communicates through optimized
protocols. Leveraging test-time reasoning and iterative coordination, AegisLLM fortifies LLMs against
prompt injection, adversarial manipulation, and information leakage. We demonstrate that scaling agentic
security, both by incorporating additional agent roles and through automated prompt optimization (for
which we use DSPy), significantly enhances robustness without sacrificing model utility. Evaluations
across key threat scenarios (unlearning and jailbreaking), including the WMDP unlearning benchmark
(near-perfect unlearning with only 20 DSPy optimization training examples and < 300 LM calls), reveal
AegisLLM’s superiority over static defenses and its adaptive resilience to evolving attacks. Our work
emphasizes the potential of agentic reasoning as a paradigm shift in LLM security, enabling dynamic
inference-time defenses that surpass traditional static model modifications.

## Project Structure

### Key Files and Directories

- `configs/`: Contains configuration files for the experiments.
    - `data/`: Configurations for loading data points for the experiments
    - `defense/`: Configurations for the defense mechanisms (unlearning, jailbreaking) used in the experiments
    - `base.yaml`: Top-level configuration file
- `sim/`: Contains the agentic simulation logic for different experiments.
- `utils/`: Logging and Litellm utility tools
- `.env`: Environment variables file to store API keys and other sensitive information. Make sure to create this file
  and add your API keys
- `requirements.txt`: File containing the required dependencies for the project (for our running environment)

### Scripts

- `optimize_dspy.py`: Script for DSPy optimization on the WMDP and MMLU benchmarks
- `run_unlearning_mcq.py`: Script for running unlearning experiments for multiple-choice questions (i.e. on the WMDP and
  MMLU benchmarks)
- `run_unlearning_mt_bench.py`: Script for running unlearning experiments for the mt_bench benchmark
- `run_unlearning_tofu.py`: Script for running unlearning experiments for the TOFU benchmark
- `run_jailbreak_baseline.py`: Script for running the jailbreak experiments using the baseline model
- `run_jailbreak_llamaguard.py`: Script for running the jailbreak experiments using the Llama-Guard model
- `run_jailbreak.py`: Script for running AegisLLM jailbreak experiments

## Setup

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Modify the `.env` file to include the required API keys for your LLM model(s).

3. Configure the `model_provider` and `model_name` values in the `configs/base.yaml` file based on the provided API
   keys.

4. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Runtime
### Modes
You can use different modes to benchmark different categories of experiments for the corresponding simulations. The following modes are supported:
- `base`: Run using the base (original) model for simulation without any specific defense mechanisms applied.
- `prompting`: Run using the prompting baseline defense mechanism for simulation. See [Guardrail Baselines for Unlearning in LLMs](https://arxiv.org/abs/2403.03329) for details.
- `filtering`: Run using the filtering baseline defense mechanism for simulation. See [Guardrail Baselines for Unlearning in LLMs](https://arxiv.org/abs/2403.03329) for details.
- `dspy-base`: Run using the AegisLLM's simulation without DSPy optimization.
- `dspy-json`: Run using the AegisLLM's simulation with DSPy optimization. You have to have run the `optimize_dspy.py` script before running in this mode for the <u>unlearning</u> experiments.

**DSPy-base** and **DSPy-json** modes correspond to our method.

## Running Experiments

### Unlearning Experiments

#### WMDP/MMLU
You must choose mode based on your experimental setup.
```sh
python3 run_unlearning_mcq.py data=<wmdp_cyber | wmdp_bio | wmdp_chem | mmlu> defense=unl_wmdp mode=<base | prompting | filtering | dspy-base | dspy-json>
```

#### MT_Bench
You must choose mode based on your experimental setup. Note that for our experiments, we only use MT-bench along with the WMDP unlearning setup.
```sh
python3 run_unlearning_mt_bench.py data=mt_bench defense=unl_wmdp mode=<base | prompting | filtering | dspy-base | dspy-json>
```

#### TOFU
You must choose mode based on your experimental setup. A limited set of modes is currently supported.
```sh
python3 run_unlearning_tofu.py data=tofu defense=unl_tofu mode=<filtering | dspy-base>
```

### Jailbreak Experiments

#### Baseline

```sh
python3 run_jailbreak_baseline.py
```

#### Llama-Guard

```sh
python3 run_jailbreak_llamaguard.py
```

#### Run_Jailbreak (Using Our Method)
You must choose mode based on your experimental setup. A limited set of modes is currently supported.
```sh
python3 run_jailbreak.py data=<false_refusal | rapid_response | strong_reject> defense=<jail_false_refusal | jail_rapid_response | jail_strong_reject> mode=<dspy-base | dspy-json>
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

[//]: # (## Citations)

[//]: # (TODO)