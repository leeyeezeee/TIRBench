#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for data cleaning experiments.
Handles configuration loading from YAML files and Sacred experiment setup.

Usage:
    python main.py with dataset_config=gsm8k agent_config=default.yaml [other parameters...]
"""

from __future__ import annotations
import os
import sys
import collections
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import yaml

# Import the run function
from run import run_experiment

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console

ex = Experiment("data_cleaning")
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = Path("results")


def _get_config(params, arg_name, subfolder):
    """Extract config file name from command line arguments."""
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        config_path = Path(__file__).parent / "config" / subfolder / f"{config_name}.yaml"
        if not config_path.exists():
            # Try without .yaml extension
            config_path = Path(__file__).parent / "config" / subfolder / config_name
            if not config_path.exists():
                return None
        
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config_dict = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"{config_name}.yaml error: {exc}")
        return config_dict
    return None


def init_config_from_yaml_files(sacred_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize configuration by merging dataset and agent YAML configs.
    This function loads YAML files and merges them into the config.
    """
    dataset_key = sacred_config.get("dataset", "gsm8k")
    agent_config_file = sacred_config.get("agent_config_file")
    
    merged_config = dict(sacred_config)
    
    # Load dataset config
    dataset_config_path = Path(__file__).parent / "config" / "dataset" / f"{dataset_key}.yaml"
    if dataset_config_path.exists():
        with open(dataset_config_path, "r", encoding="utf-8") as f:
            dataset_config = yaml.safe_load(f) or {}
        merged_config["dataset_config"] = dataset_config
        # Also merge at top level for backward compatibility
        if dataset_config.get("default_source"):
            merged_config["source"] = dataset_config["default_source"]
        if dataset_config.get("default_input"):
            merged_config["input"] = os.path.expandvars(dataset_config["default_input"])
    
    # Load agent config
    if agent_config_file:
        agent_config_path = Path(__file__).parent / "config" / "agent" / agent_config_file
        if not agent_config_path.exists():
            agent_config_path = Path(__file__).parent / "config" / "agent" / "default.yaml"
    else:
        agent_config_path = Path(__file__).parent / "config" / "agent" / "default.yaml"
    
    if agent_config_path.exists():
        with open(agent_config_path, "r", encoding="utf-8") as f:
            agent_config = yaml.safe_load(f) or {}
        merged_config["agent_config"] = agent_config
        # Also merge at top level for backward compatibility
        if agent_config.get("backend"):
            merged_config["backend"] = agent_config["backend"]
        if agent_config.get("model_path"):
            merged_config["model"] = agent_config["model_path"]
        if agent_config.get("tokenizer_path"):
            merged_config["tokenizer_path"] = agent_config["tokenizer_path"]
        
        # Merge generation params
        if agent_config.get("generation"):
            gen = agent_config["generation"]
            if "temperature" in gen:
                merged_config["temperature"] = gen["temperature"]
            if "top_p" in gen:
                merged_config["top_p"] = gen["top_p"]
            if "max_tokens" in gen:
                merged_config["max_input_tokens"] = gen["max_tokens"]
            if "max_new_tokens" in gen:
                merged_config["max_new_tokens"] = gen["max_new_tokens"]
        
        # Merge vLLM config
        if agent_config.get("vllm"):
            vllm_config = agent_config["vllm"]
            if "tensor_parallel_size" in vllm_config:
                merged_config["tp"] = vllm_config["tensor_parallel_size"]
            if "gpu_memory_utilization" in vllm_config:
                merged_config["gpu_memory_utilization"] = vllm_config["gpu_memory_utilization"]
    
    return merged_config


def recursive_dict_update(d, u):
    """Recursively update dictionary d with values from u."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    """Deep copy configuration."""
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


@ex.config
def default_config():
    """Default configuration for Sacred experiment."""
    # Dataset configuration
    dataset = "gsm8k"
    source = "json"
    input_path = None
    split = "validation"
    sample_size = 500
    max_eval_examples = 0
    seed = 42
    store_think = False
    store_prompt = False
    store_context = 0
    run_tag = ""
    decision = "auto"
    squad_f1_threshold = 0.8
    hotpot_f1_threshold = 0.8
    nq_f1_threshold = 0.8
    results_dir = "results"
    save_original = False
    
    # Backend configuration (will be merged with agent config)
    backend = "vllm"
    model = ""
    tokenizer_path = None
    local_files_only = False
    use_chat_template = False
    temperature = 0.0
    top_p = 1.0
    max_input_tokens = 2048
    max_new_tokens = 64
    
    # vLLM
    tp = 1
    gpu_memory_utilization = 0.9
    
    # SGLang
    sglang_api_base = os.getenv("SGLANG_API_BASE", "http://127.0.0.1:30000")
    sglang_api_key = os.getenv("SGLANG_API_KEY", None)
    sglang_model = None
    concurrency = 8
    
    # Transformers
    device = "cuda"
    device_map = None
    
    # Filtering & outputs
    batch_size = 8
    f1_threshold = 0.8
    export_squad_like = False
    output = ""
    save_csv = None
    no_logs = False
    eval_only = False
    
    # Dataset hints
    include_unanswerable_hint = False
    handle_unanswerable = False
    hotpot_config = "distractor"
    
    # Config file paths
    agent_config_file = None  # Path to agent config YAML (relative to config/agent/)
    use_agent_tools = False  # Whether to use LLMAgent with tools
    
    # Dataset and agent configs (loaded from YAML files)
    dataset_config = {}
    agent_config = {}


@ex.main
def main(_run, _config, _log):
    """Sacred main function that calls the experiment runner."""
    config = config_copy(_config)
    return run_experiment(_run, config, _log)


if __name__ == "__main__":
    params = deepcopy(sys.argv)
    
    # Get default config (if exists)
    default_config_path = Path(__file__).parent / "config" / "default.yaml"
    if default_config_path.exists():
        with open(default_config_path, "r", encoding="utf-8") as f:
            try:
                config_dict = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"default.yaml error: {exc}")
    else:
        config_dict = {}
    
    # Extract dataset and agent config file names from command line
    # Format: --dataset-config=gsm8k or --agent-config=default.yaml
    dataset_config_file = None
    agent_config_file = None
    
    for i, param in enumerate(params):
        if param.startswith("--dataset-config="):
            dataset_config_file = param.split("=", 1)[1]
            params[i] = None  # Mark for removal
        elif param.startswith("--agent-config="):
            agent_config_file = param.split("=", 1)[1]
            params[i] = None  # Mark for removal
    
    # Remove None entries
    params = [p for p in params if p is not None]
    
    # Get config from argv (command line overrides)
    def _get_argv_config(params):
        config = {}
        to_del = []
        for _i, _v in enumerate(params):
            if "=" not in _v:
                continue
            item = _v.split("=")[0]
            if item[:2] == "--":
                config_v = _v.split("=")[1]
                try:
                    config_v = eval(config_v)
                except:
                    pass
                # Convert --param-name to param_name
                param_name = item[2:].replace("-", "_")
                config[param_name] = config_v
                to_del.append(_v)
        for _v in to_del:
            if _v in params:
                params.remove(_v)
        return config
    
    # Merge command line overrides
    config_dict = recursive_dict_update(config_dict, _get_argv_config(params))
    
    # Set agent_config_file if specified
    if agent_config_file:
        config_dict["agent_config_file"] = agent_config_file
    
    # Now load and merge YAML configs (dataset config is loaded based on dataset key)
    config_dict = init_config_from_yaml_files(config_dict)
    
    # Add all config to Sacred
    ex.add_config(config_dict)
    
    # Setup Sacred observers
    results_dir = Path("results") / "sacred"
    results_dir.mkdir(parents=True, exist_ok=True)
    ex.observers.append(FileStorageObserver.create(str(results_dir)))
    
    # Run Sacred experiment
    ex.run_commandline(params)

