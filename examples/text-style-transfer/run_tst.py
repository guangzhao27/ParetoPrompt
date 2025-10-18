import os
import dataclasses
import hydra
from hydra.core.config_store import ConfigStore
from typing import Optional
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('../../')
from trainers import TrainerConfig, make_trainer
from modules import SQLModuleConfig, make_sql_module
from models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model)
from utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir, 
                                  algorithm_set_config,
                                  )
from tst_helpers import (PromptedTextStyleTransferRewardConfig,
                         TextStyleTransferDatasetConfig,
                         make_prompted_text_style_transfer_reward,
                         make_text_style_transfer_datasets,
                         get_style_classifier, 
                         )
from dataclasses import dataclass

import random
import numpy as np
import torch

# import os
# os.environ['HF_HOME'] = './llm_cache_dir'


@dataclass
class LoadConfig:
    '''
    objective types: 
        two: content and positiveness
        three1: content, positiveness and prompt fluency
        three2: content, positiveness and output fluency
    '''
    model_path: Optional[str] = None
    algorithm_name: str="RlPrompt"
    load_step: int = 0
    few_shot: int = -1
    dominate_evaluate_num: int = 16
    temperature: float = 0.3
    objective_type: str='two'
    task_type: str='tst'

# Compose default config
config_list = [PromptedTextStyleTransferRewardConfig,
                TextStyleTransferDatasetConfig, LMAdaptorModelConfig,
                SinglePromptModelConfig, SQLModuleConfig, TrainerConfig, LoadConfig]
cs = compose_hydra_config_store('base_tst', config_list)


@hydra.main(version_base=None, config_path="./configs", config_name="tst_config")
def main(config: "DictConfig"):
    
    config.prompt_train_batch_size = config.num_repeats*config.train_batch_size ## TODO: this should be fixed, too many free configs
    
    output_dir = get_hydra_output_dir()

    train_dataset, val_dataset, test_dataset = \
        make_text_style_transfer_datasets(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])

    print("Algorithm Name: ", config.algorithm_name)
    algorithm_set_config(config)
    
    colorful_print(OmegaConf.to_yaml(config), fg='red')

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    config.style_classifier = get_style_classifier('train', config)
    reward = make_prompted_text_style_transfer_reward(config) # reward top_k decide the text generation, set as 1 for deterministic generation
    algo_module = make_sql_module(prompt_model, reward, config) #module top_k decide the prompt generation, set as 0 for diverse generation

    config.save_dir = os.path.join(output_dir, config.save_dir)
    if config.run_name:
        config.run_name = output_dir+config.run_name
    else:
        config.run_name = output_dir
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    
    if config.model_path:
        load_ckpt_path = os.path.join('./',
            config.model_path, 
            f"outputs/ckpt/ckpt.step.{config.load_step}.pth")
        trainer.load_pretrain(load_ckpt_path, config.training_device)
        reward._counter=config.load_step
    
    trainer.train(config=config)


if __name__ == "__main__":
    main()