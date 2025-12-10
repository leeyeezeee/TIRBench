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
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ValueError(f"{config_name}.yaml error: {exc}")
        return config_dict
    return None


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


@ex.main
def main(_run, _config, _log):
    """Sacred main function that calls the experiment runner."""
    config = config_copy(_config)
    return run_experiment(_run, config, _log)


if __name__ == "__main__":
    params = deepcopy(sys.argv)
    
    # Load default config
    default_config_path = Path(__file__).parent / "config" / "default.yaml"
    with open(default_config_path, "r", encoding="utf-8") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"default.yaml error: {exc}")
    
    # Override with environment variables if present
    if os.getenv("SGLANG_API_BASE"):
        config_dict["sglang_api_base"] = os.getenv("SGLANG_API_BASE")
    if os.getenv("SGLANG_API_KEY"):
        config_dict["sglang_api_key"] = os.getenv("SGLANG_API_KEY")
    
    # Load dataset and agent configs from command line
    dataset_config = _get_config(params, "--dataset-config", "dataset")
    agent_config = _get_config(params, "--agent-config", "agent")
    
    # Merge configs: default -> dataset -> agent
    if dataset_config:
        config_dict = recursive_dict_update(config_dict, dataset_config)
    if agent_config:
        config_dict = recursive_dict_update(config_dict, agent_config)
    
    # Add all config to Sacred
    ex.add_config(config_dict)
    
    # Setup Sacred observers
    results_dir = Path("results") / "sacred"
    results_dir.mkdir(parents=True, exist_ok=True)
    ex.observers.append(FileStorageObserver.create(str(results_dir)))
    
    # Run Sacred experiment
    ex.run_commandline(params)

