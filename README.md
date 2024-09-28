# POEX: Policy Executable Embodied AI Jailbreak Attacks

## Anonymous Authors

[project page](https://usenix2025-1-167.github.io/)


## Installation Guide

```
git clone https://github.com/USENIX2025-1-167/POEX
cd POEX
pip install -r requirements.txt
pip install -U git+https://github.com/lm-sys/FastChat
```


## Running the Code

### Downloading the model

Download the open-source LLM into the `models` folder.
Put the lora model into the `models` folder.

### Setting Your OpenAI API Key

Before running the code, make sure to set your OpenAI API key as an environment variable, or manually set the `openai.api_key` in `main.py`:
```
export OPENAI_API_KEY = your openai api key
```
We provide openai api key in openaiapikey.png

### Starting the Training

The following command will start to train adversarial suffixes on harmful-rlbench
```
python main.py
```

The available arguments and their options are as follows:
- `--attack_model_name`: the open-source LLM name.
- `--attack_model_path`: the open-source LLM path.
- `--policy_evaluator`: `llama3` or `openai`.
- `--output_path`: result save path.



Thanks to Easyjailbreak!
