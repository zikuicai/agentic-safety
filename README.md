# AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security

We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications.
 
## Project Structure

### Key Files and Directories

- `configs/`: Contains configuration files for the experiments.
    - `data/`: Configurations for loading data points for the experiments
    - `defense/`: Configurations for the defense mechanisms (unlearning, jailbreaking) used in the experiments
    - `base_unl.yaml`: Top-level configuration file for unlearning experiments
    - `base_jail.yaml`: Top-level configuration file for jailbreak experiments
- `pipelines/`: Contains the agentic pipelines for different agentic architectures.
- `utils/`: Logging and Litellm utility tools
- `.env`: Environment variables file to store API keys and other sensitive information. Make sure to create this file
  and add your API keys
- `requirements.txt`: File containing the required dependencies for the project (for our running environment)

### Scripts

- `optimize_unlearning_dspy.py`: Script for DSPy optimization on the WMDP and MMLU benchmarks for unlearning
- `optimize_jailbreak_dspy.py`: Script for DSPy optimization on the the StrongReject and FalseRefusal benchmarks for
  jailbreaking
- Unlearning Runs:
  - `run_unlearning_mcq.py`: Script for running unlearning experiments for multiple-choice questions (i.e. on the WMDP and
    MMLU benchmarks)
  - `run_unlearning_mt_bench.py`: Script for running unlearning experiments for the mt_bench benchmark
  - `run_unlearning_tofu.py`: Script for running unlearning experiments for the TOFU benchmark
- Jailbreak Runs:
  - `run_jailbreak_baseline.py`: Script for running the jailbreak experiments using the baseline model
  - `run_jailbreak_llamaguard.py`: Script for running the jailbreak experiments using the Llama-Guard model
  - `run_jailbreak_dspy.py`: Script for running jailbreak experiments using the AegisLLM pipeline (optimized with DSPy)

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

## Experimentation

### Modes

You can use different modes to benchmark different categories of experiments for the corresponding pipelines. The
following modes are supported:

- `base`: Run using the base (original) model for pipeline without any specific defense mechanisms applied.
- `prompting`: Run using the prompting baseline defense mechanism for pipeline.
  See [Guardrail Baselines for Unlearning in LLMs](https://arxiv.org/abs/2403.03329) for details.
- `filtering`: Run using the filtering baseline defense mechanism for pipeline.
  See [Guardrail Baselines for Unlearning in LLMs](https://arxiv.org/abs/2403.03329) for details.
- `dspy-base`: Run using the AegisLLM's pipeline **without** DSPy optimization.
- `dspy-json`: Run using the AegisLLM's pipeline **with** DSPy optimization. You have to have run the `optimize_unlearning_dspy.py` and `optimize_jailbreak_dspy.py`
  scripts before running in this mode for the <u>unlearning</u> and <u>jailbreaking</u> experiments, respectively.

**DSPy-base** and **DSPy-json** modes correspond to our method.

## Running Experiments

### Unlearning Experiments
**Note:** To run any unlearning experiments in the `dspy-json` mode (WMDP/MMLU and MT-Bench), you have to first optimize the DSPy pipeline (Orchestrator only) using the following script:

```sh
python3 optimize_unlearning_dspy.py +data=wmdp_cyber +defense=unl_wmdp
```
The resulting optimization will be saved in the `dspy_optimized_dir` directory specified in `configs/base_unl.yaml`. You only need to run this optimization once for any WMDP/MMLU, and MT-Bench experiments, and can use it without limitations afterwards. 
#### WMDP/MMLU

You must choose mode based on your experimental setup.

```sh
python3 run_unlearning_mcq.py +data=<wmdp_cyber | wmdp_bio | wmdp_chem | mmlu> +defense=unl_wmdp mode=<base | prompting | filtering | dspy-base | dspy-json>
```
**Note:** Please also check out the file `run_unl_mcq.sh` for an example of how this script can be used in practice in a bash-script. 
#### MT_Bench

You must choose mode based on your experimental setup. Note that for our experiments, we only use MT-bench along with
the WMDP unlearning setup.

```sh
python3 run_unlearning_mt_bench.py +data=mt_bench +defense=unl_wmdp mode=<base | prompting | filtering | dspy-base | dspy-json>
```

#### TOFU

You must choose mode based on your experimental setup. A limited set of modes is currently supported.

```sh
python3 run_unlearning_tofu.py +data=tofu +defense=unl_tofu mode=<filtering | dspy-base> data.data.subset=<forget01 | forget05 | forget10>
```

### Jailbreak Experiments

For the jailbreak experiments, you need to first run Python scripts to generate the data points, which will later be _evaluated_ (see [Jailbreak Defense Evaluations](#jailbreak-defense-evaluations)). 

#### Jailbreak Defenses
You can use one of the following methods (each method supports :

1. **Baseline**

```sh
python3 run_jailbreak_baseline.py +data=<false_refusal | strong_reject> +defense=<jail_false_refusal | jail_strong_reject>
```

2. **Llama-Guard**

```sh
python3 run_jailbreak_llamaguard.py +data=<false_refusal | strong_reject> +defense=<jail_false_refusal | jail_strong_reject>
```

3. **Jailbreak (AegisLLM):**

**Note:** To run this experiment in the `dspy-json` mode, you have to first optimize the DSPy pipeline (Orchestrator and Evaluator) using the following script:

```sh
python3 optimize_jailbreak_dspy.py +data=<false_refusal | strong_reject> +defense=<jail_false_refusal | jail_strong_reject>
```
Currently, both options for `data` and `defense` perform the same optimization, and you can choose either one. You would only need to run the optimization once. The optimized pipeline will be saved in the `dspy_optimized_dir` directory specified in `configs/base_jail.yaml`. For future developments, such optimization configs could be separated for each data and defense combination.

After any optimizations are done, to run this experiment, you can run the following script. A limited set of modes is currently supported.
```sh
python3 run_jailbreak_dspy.py +data=<false_refusal | strong_reject> +defense=<jail_false_refusal | jail_strong_reject> mode=<dspy-base | dspy-json>
```
#### Jailbreak Defense Evaluations
After running the false_refusal or strong_reject Python scripts for the jailbreak experiments, you can evaluate the results using the following scripts for each of the corresponding benchmarks. Each of the scripts in the [Jailbreak Defenses](#jailbreak-defenses) section will provide you with an output JSON/JSONl file path, which you can use as input to the evaluation scripts in the following scripts.

1. False Refusals (PHTest)
For this script, you need to provide the input path of the JSON/JSONl file generated by experiments run, the judge model names and API access information, and an arbitrary output path which would be used to save the resulting evaluation information. You will be provided with _false_refusal_ scores in your shell after the run is complete. Please note that the script supports extra input parameters that you can check out using the `--help` flag.

```sh
python3 evals/false_refusal/eval_false_refusal.py --input-path <input_path> --output-path <arbitrary_output_path> --model-name <model_name>
```

2. Strong Reject
Similar to the script for evaluating False Refusals, you need to provide the input path of the JSON/JSONl file generated by experiments run and an arbitrary output path which would be used to save the resulting evaluation information for the evaluations script for Strong Reject. You will be provided with _strong_reject_ scores in your shell after the run is complete. Please note that the script supports extra input parameters that you can check out using the `--help` flag.

```sh
python3 evals/strong_reject/eval_strong_reject.py --input-path <input_path> --output-path <arbitrary_output_path>
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
