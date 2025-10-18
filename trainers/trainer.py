import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, Union, List, Tuple
import os
import wandb
import json
import click
import torch.nn.functional as F
import random
from modules import BaseModule
from utils import utils
from .trainer_utils import get_default_train_op, set_random_seed, find_pareto_front, calculate_dominating_volume, evaluate_model_dominate_volume
import copy
from collections import defaultdict
import matplotlib.pyplot as plt

EVAL_STEPS_fsc = [1, 2, 200, 400, 600, 800, 1000, 2000, 4000, 6000]
EVAL_STEPS_tst = [1, 200, 500, 1000, 2000, 4000,  5000, 10000]
class Trainer:
    """Trainer that runs for a specified number of epochs. 

    Each epoch can run for a specified number of batches.
    Evaluation is done at the end of each epoch """

    def __init__(
        self,
        module: BaseModule,
        # Train params
        train_dataset: Optional[Dataset],
        train_batch_size: int,
        train_shuffle: bool,
        train_drop_last: bool,
        num_train_epochs: int,
        max_train_steps: int,
        # Eval params
        do_eval: bool,
        eval_dataset: Optional[Dataset],
        eval_batch_size: int,
        eval_steps: int,
        # Save params
        do_save: bool,
        save_dir: str,
        save_steps: int,
        # Optimizer params
        learning_rate: float,
        gradient_clip: bool,
        gradient_clip_norm: float,
        # Checkpoint params
        checkpoint_path: Optional[str],
        # Random seed
        random_seed: Optional[int],
        # Wandb reporting
        report_to_wandb: bool,
        project_name: Optional[str],
        run_name: Optional[str]
    ):
        assert do_eval == False or eval_dataset is not None, \
            "Need to have eval_dataset if do_eval is True"
        self.module = module

        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.train_drop_last = train_drop_last
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps

        self.do_eval = do_eval
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps

        self.do_save = do_save
        self.save_dir = save_dir
        self.save_steps = save_steps
        
        self.optimizer = torch.optim.Adam(self.module._model.parameters(),
                           lr=learning_rate)

        self.train_op = get_default_train_op(self.module._model,
                                             learning_rate,
                                             gradient_clip,
                                             gradient_clip_norm)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        if random_seed is not None:
            set_random_seed(random_seed)

        self.report_to_wandb = report_to_wandb
        self.project_name = project_name
        self.run_name = run_name
        self.eval_num = 0
        self.pareto_front_dict = {}

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.module.load_state_dict(checkpoint["model_state_dict"])
        print(click.style(f"Loaded module from {checkpoint_path}", fg="green"))

    def _get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          shuffle=self.train_shuffle,
                          batch_size=self.train_batch_size,
                          drop_last=self.train_drop_last)

    # @torch.no_grad
    def _train_step(
        self,
        step: int,
        batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        model = self.module.train()
        model._pre_steps(step)

        loss, batch_log = model(batch)
        loss.backward()
        batch_log['SQL_ON/rewards/loss'] = loss.item()
        self.optimizer.step()
        self.optimizer.zero_grad()
        optimizer_state = self.optimizer.state_dict()
        # optimize_state = self.train_op()
        
        # if self.report_to_wandb:
        #     wandb.log(batch_log, step=step)

        return batch_log, optimizer_state

    def train(self,
              report_to_wandb: Optional[bool] = None,
              project_name: Optional[str] = None,
              run_name: Optional[str] = None,
              config: Optional["DictConfig"] = None, 
              reward2=None) -> None:
        # Configure Wandb reporting
        if report_to_wandb is None:
            report_to_wandb = self.report_to_wandb
        if project_name is None:
            project_name = self.project_name
        if run_name is None: 
            run_name = self.run_name
        # if config is not None: 
        #     config = eval(str(config))
        if report_to_wandb:
            os.environ["WANDB_API_KEY"] = "e48870a815eed7ccfabbb6d1a0e40f9a618bfa88"
            wandb.init(project=project_name, name=run_name, config=eval(str(config)))
            wandb.watch(self.module, log=None)
            print(wandb.run.dir)
            print(self.run_name)
        
        self.eval_steps = EVAL_STEPS_tst if config.task_type == 'tst' else EVAL_STEPS_fsc
        self.different_reward=reward2
        
        # Create saving path
        eval_save_dir = os.path.join(self.save_dir, "eval")
        ckpt_save_dir = os.path.join(self.save_dir, "ckpt")
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        train_dataloader = self._get_train_dataloader()
        # Determine whether to train by epoch or steps
        if self.max_train_steps < 0:
            total_train_epochs = self.num_train_epochs
        else:
            num_batches_per_epoch = len(train_dataloader)
            total_train_epochs = \
                (self.max_train_steps // num_batches_per_epoch
                 + int(self.max_train_steps % num_batches_per_epoch > 0))

        # Determine whether to evaluate by epoch or steps
        eval_by_steps = True # self.eval_steps > 0
        # Determine whether to save by epoch or steps
        save_by_steps = self.save_steps > 0

        total_steps = 0
        for epoch in range(total_train_epochs):
            for step, batch in enumerate(train_dataloader):
                total_steps += 1
                if config.model_path and step < config.load_step:
                    continue
                batch_log, _ = self._train_step(total_steps, batch)
                if report_to_wandb:
                    wandb.log(batch_log, step=total_steps) # The loss is calculated by averaging the loss of generated prompts from the prmopt modle 
                

                if self.do_eval and eval_by_steps \
                        and total_steps in self.eval_steps:
                    print('start evaluation')
                    output_save_path = \
                        os.path.join(eval_save_dir,
                                     f'outputs.step.{total_steps}.json')
                    eval_log, performance = self.evaluate(output_save_path=output_save_path, 
                                                          dominate_evaluate_num=config.dominate_evaluate_num, 
                                                          run_name = run_name,
                                                          total_steps=total_steps
                                                          )
                    if report_to_wandb:
                        wandb.log(eval_log, step=total_steps)
                    
                    # file_name = run_name + '.json'
                    # if os.path.exists(file_name):
                    #     with open(file_name, 'r') as f:
                    #         performance_dict = json.load(f)
                    # else:
                    #     performance_dict = {}
                    # performance_dict[total_steps] = performance
                    
                    # with open(file_name, 'w') as f:
                    #     json.dump(performance_dict, f)

                if self.do_save and save_by_steps \
                        and total_steps in self.eval_steps:
                    torch.save({"steps": total_steps,
                                "model_state_dict": self.module.state_dict(), 
                                "optimizer_state": self.optimizer.state_dict(), 
                                },
                               os.path.join(ckpt_save_dir,
                                            f"ckpt.step.{total_steps}.pth"))

                if total_steps == self.max_train_steps:
                    break

            if self.do_eval and not eval_by_steps:
                output_save_path = os.path.join(eval_save_dir,
                                                f'outputs.epoch.{epoch+1}.json')
                eval_log, performance = self.evaluate(output_save_path=output_save_path)
                wandb.log(eval_log, step=total_steps)

            if self.do_save and not save_by_steps:
                torch.save({"steps": total_steps,
                            "model_state_dict": self.module.state_dict()},
                           os.path.join(ckpt_save_dir,
                                        f"ckpt.epoch.{epoch+1}.pth"))

    def _get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
        return DataLoader(eval_dataset,
                          batch_size=self.eval_batch_size)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        output_save_path: Optional[str] = None,
        dominate_evaluate_num: int=16,
        compute_scores: bool = True,
        run_name = None,
        total_steps = 0, 
    ) -> Dict[str, np.number]:
        
        self.eval_num += 1
        
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_dataloader = self._get_eval_dataloader(eval_dataset)

        model = self.module.eval()
        hypos = []
        scores: List[List[str]] = []
        for batch in eval_dataloader:
            infer_outputs: Dict[str, Union[torch.Tensor, List[List[str]]]]
            infer_outputs = model.infer(batch)

        #these three lines can be skipped. this one calculate the best token performance. While dominate volume calculate all the tokens performance
        # hypos += infer_outputs['sample_tokens'] #hypos are the most likely prompts
        score, multi_rewards_dict, tokens_list, score_log = model.compute_rewards(
            batch=batch,
            output_tokens=infer_outputs['sample_tokens'], multi_optimize=True)
        scores += score.detach().tolist()
    
        # generate many prompt like load and log the mean style and content and calculate the dominating area]
        output_tokens_P = list(self.pareto_front_dict.keys()) if self.pareto_front_dict is not None else None
        dominate_volume, performance = evaluate_model_dominate_volume(model=model, batch=batch, dominate_evaluate_num=dominate_evaluate_num, 
                                                                          output_tokens_P=output_tokens_P)
        # dominate_volume, performance = evaluate_model_dominate_volume(model=model, batch=batch, dominate_evaluate_num=dominate_evaluate_num,
        #                                                             )
        if self.different_reward:
            dominate_volume2, performance2 = evaluate_model_dominate_volume(model=model, batch=batch, dominate_evaluate_num=dominate_evaluate_num,
                                                                            output_tokens_P=output_tokens_P,
                                                                            different_reward=self.different_reward)
        else:
            dominate_volume2=0
        # if output_save_path is not None:
        #     json.dump({'output_tokens': hypos,
        #                'scores': scores},
        #               open(output_save_path, 'w'))
        
        # object1_name, object2_name = multi_rewards_dict.keys()
        # object1_value, object2_value = multi_rewards_dict[object1_name], multi_rewards_dict[object2_name]
        
        # score = score.mean().item()
        # object1_value = object1_value.mean().item()
        # object2_value = object2_value.mean().item()
        objects_name_list = list(multi_rewards_dict.keys())
        object1_name = objects_name_list[0]
        object2_name = objects_name_list[1]
        object1_value = torch.tensor(performance[object1_name]).max().item()
        object2_value = torch.tensor(performance[object2_name]).max().item()
        performance['dominating_volume2'] = dominate_volume2
        # performance['pareto_front'] = self.pareto_front_dict
        utils.add_prefix_to_dict_keys_inplace(
            score_log,
            prefix=f"eval/rewards/")

        print('Finish Eval')
        
        #save performance data to json
        file_name = run_name + '.json'
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                performance_dict = json.load(f)
        else:
            performance_dict = {}
        performance_dict[total_steps] = performance
        
        with open(file_name, 'w') as f:
            json.dump(performance_dict, f)
            
        #draw scatter
        
        if total_steps in self.eval_steps:
            plt.scatter(performance[object1_name], performance[object2_name], label=total_steps)
            plt.xlabel(f'{object1_name} score')
            plt.ylabel(f'{object2_name} score')
            plt.legend()
            plt.savefig(run_name+str(total_steps)+'.png', format='png')
            plt.figure()
        
        return utils.unionize_dicts([
            score_log,
            # gem_scores_dict,
            {
                f"eval/dominate_volume": dominate_volume,
                f"eval/dominate_volume2": dominate_volume2, 
                f"eval/pareto_size": performance['pareto_size'],
                f"eval/score": score,
                f"eval/{object1_name}":object1_value, 
                f"eval/{object2_name}": object2_value, 
                f"eval/output_length": np.mean([len(tokens) \
                                                for tokens in hypos])
            }
        ]), performance
        
    def load_pretrain(self, ckpt_path, training_device):
        # check point keys ['steps', 'model_state_dict', 'optimizer_state']
        if training_device == "cpu":
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        elif training_device == "cuda":
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cuda"))
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.module.load_state_dict(checkpoint['model_state_dict'])
        
    def update_pareto_front(self, multi_rewards_dict, tokens_list):
        tokens_list += list(self.pareto_front_dict.keys())
        combined_rewards = torch.stack(list(multi_rewards_dict.values()), dim=1)
        
        if self.pareto_front_dict:
            old_pareto_front = torch.stack(list(self.pareto_front_dict.values()))
            combined_rewards = torch.cat((combined_rewards, old_pareto_front))
            
        pareto_front_indices = []
        dominated_by = torch.zeros(combined_rewards.size(0), dtype=torch.bool)
        
        for idx, point1 in enumerate(combined_rewards):
            if not dominated_by[idx]:
                pareto_front_indices.append(idx)
                for j, point2 in enumerate(combined_rewards):
                    if idx != j:
                        if torch.all(point1 >= point2):
                            dominated_by[j] = True
                        elif torch.all(point1 <= point2):
                            dominated_by[idx] = True
                            pareto_front_indices.remove(idx)
                            break
        # new_pareto_front_dict = {}
        # for idx in pareto_front_indices:
        #     token = tuple(tokens_list[idx])
        #     tensor = combined_rewards[idx]
        #     new_pareto_front_dict[token] = tensor
        # self.pareto_front_dict = new_pareto_front_dict
        self.pareto_front_dict = {tuple(tokens_list[idx]): combined_rewards[idx] for idx in pareto_front_indices}
        