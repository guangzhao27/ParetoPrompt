import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, Union, List, Tuple
import torch.nn.functional as F
import random
from modules import BaseModule
from utils import utils
from .trainer_utils import get_default_train_op, set_random_seed, find_pareto_front, calculate_dominating_volume, evaluate_model_dominate_volume

from trainers import Trainer
    


def reward_diff(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios
    return logits


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    dpo_loss_config) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    name = dpo_loss_config.name
    beta = dpo_loss_config.beta
    # reference_free = dpo_loss_config.reference_free
    label_smoothing = dpo_loss_config.label_smoothing

    logits = reward_diff(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
    # also known as h_{\pi_\theta}^{y_w,y_l}

    if name == "ipo":
        losses = (logits - 1/(beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    elif name == "dpo":
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    else:
        raise ValueError(f'unknown loss {name}')

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


class DPO_Trainer(Trainer):
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
        run_name: Optional[str],
        dpo_loss_config: Dict[str, Any],
    ):
        super().__init__(
            module,
            train_dataset,
            train_batch_size,
            train_shuffle,
            train_drop_last,
            num_train_epochs,
            max_train_steps,
            do_eval,
            eval_dataset,
            eval_batch_size,
            eval_steps,
            do_save,
            save_dir,
            save_steps,
            learning_rate,
            gradient_clip,
            gradient_clip_norm,
            checkpoint_path,
            random_seed,
            report_to_wandb,
            project_name,
            run_name, 
        )
        self.dpo_loss_config = dpo_loss_config
        
    def _train_step(self, step: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        policy_model = self.module.train()
        policy_model._pre_steps(step)         
        
        (policy_logits, ref_logits, output_tokens, output_ids, sequence_lengths) = \
            policy_model._decode_sampling(batch=batch)  ## TODO: where should these samples being sampled from??
            
        raw_rewards, rewards_log = \
            policy_model.compute_rewards(batch=batch, 
                                output_tokens=output_tokens,
                                mode="train")
        
        reward_matrix = policy_model._reward_shaping_func(raw_rewards)
        sample_num = reward_matrix.shape[0]
        token_lenght = output_ids.shape[1]
        
        policy_all_logps = policy_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        
        # ref_model = copy.deepcopy(policy_model).eval()
        # ref_model._pre_steps(step)
        # outputs_ = ref_model.teacher_forcing(**batch)  ## what is ** batch, check if ref_model update, follow the debuuging process s
        
        # ref_logits = outputs_['sample_logits']
        if self.dpo_loss_config.name == 'reinforce':
            log_prob = F.log_softmax(policy_logits, dim=-1)
            policy_all_logps = log_prob[
                torch.arange(sample_num).unsqueeze(1), 
                torch.arange(token_lenght), 
                output_ids
            ].sum(axis=1)
            losses = -policy_all_logps*raw_rewards
        else:
            ref_all_logps = ref_logits[
                torch.arange(sample_num).unsqueeze(1), 
                torch.arange(token_lenght), 
                output_ids
            ].sum(axis=1)
            
            
            chosen_idx_list = torch.zeros(sample_num // 2, dtype=torch.long)
            rejected_idx_list = torch.zeros(sample_num // 2, dtype=torch.long)
            for i in range(sample_num // 2):
                if reward_matrix[2*i] > reward_matrix[2*i + 1]:
                    chosen_idx_list[i] = 2 * i
                    rejected_idx_list[i] = 2 * i + 1
                else:
                    chosen_idx_list[i] = 2 * i + 1
                    rejected_idx_list[i] = 2 * i## make it random?
            
            
            
            policy_chosen_logps = policy_all_logps[chosen_idx_list]
            policy_rejected_logps = policy_all_logps[rejected_idx_list]
            with torch.no_grad():
                reference_chosen_logps = ref_all_logps[chosen_idx_list]
                reference_rejected_logps = ref_all_logps[rejected_idx_list]
            
            if self.dpo_loss_config.name == 'dpo':
                    loss_kwargs = {'beta': self.dpo_loss_config.beta, 'reference_free': self.dpo_loss_config.reference_free, 'label_smoothing': self.dpo_loss_config.label_smoothing, 'ipo': False}
            elif self.dpo_loss_config.name == 'ipo':
                loss_kwargs = {'beta': self.dpo_loss_config.beta, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {self.dpo_loss_config.name}')

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, self.dpo_loss_config)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        loss = losses.mean()
        loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        optimizer_state = self.optimizer.state_dict()
        
        # optimizer_state = self.train_op()
        
        batch_log = rewards_log  ## add more recorded metrics in this log
        
        utils.add_prefix_to_dict_keys_inplace(
                rewards_log, prefix="SQL_ON/rewards/")
        
        batch_log = rewards_log
        batch_log['dpo_loss'] = loss

        return batch_log, optimizer_state
       
       
class DPO_O2_Trainer(Trainer):
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
        run_name: Optional[str],
        dpo_loss_config: Dict[str, Any],
    ):
        super().__init__(
            module,
            train_dataset,
            train_batch_size,
            train_shuffle,
            train_drop_last,
            num_train_epochs,
            max_train_steps,
            do_eval,
            eval_dataset,
            eval_batch_size,
            eval_steps,
            do_save,
            save_dir,
            save_steps,
            learning_rate,
            gradient_clip,
            gradient_clip_norm,
            checkpoint_path,
            random_seed,
            report_to_wandb,
            project_name,
            run_name, 
        )
        self.dpo_loss_config = dpo_loss_config
        
    def loss_func(self, step, batch, config):
        policy_model = self.module.train()
        policy_model._pre_steps(step)  
        
        (policy_logits, ref_logits, output_tokens, output_ids, sequence_lengths) = \
            policy_model._decode_sampling(batch=batch) # for multiobjective batch size should be large
        
        sample_num = policy_logits.shape[0]
        token_lenght = policy_logits.shape[1]
        
        policy_all_logps = policy_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        
        ref_all_logps = ref_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        return policy_all_logps, ref_all_logps
        

    def _train_step(self, step: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        
        loss, rewards_log = self.dpo_loss_calculation(step, batch, self.dpo_loss_config)
        
        loss.backward()#0.4288172721862793
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        optimizer_state = self.optimizer.state_dict()
        # optimizer_state = self.train_op()
        
        # batch_log = rewards_log  ## add more recorded metrics in this log
        utils.add_prefix_to_dict_keys_inplace(
                rewards_log, prefix="SQL_ON/rewards/")
        
        batch_log = rewards_log
        batch_log['dpo_loss'] = loss

        return batch_log, optimizer_state
    
    def dpo_loss_calculation(self, step, batch, dpo_loss_config):

        

        policy_model = self.module.train()
        policy_model._pre_steps(step)  
        
        sum_reward, multi_rewards_dict, rewards_log, policy_all_logps, ref_all_logps = \
        self.sample_prompt_calculate_logp(batch, policy_model)
        
        # assert len(multi_rewards_dict.keys()) == 2, 'the key number must be 2'
        object_values = torch.stack(list(multi_rewards_dict.values()), axis=1)
        
        sample_num = len(sum_reward)
        idxs = list(range(sample_num))
        random.shuffle(idxs)
        idx_pairs = [(idxs[i], idxs[i+1]) for i in range(0, sample_num, 2)]
        idx_pairs = [(i, j) for i in range(sample_num) for j in range(i+1, sample_num)]
        
        chosen_idx_list = []
        rejected_idx_list = []
        non_dominate_list = [[], []]

        def log_extract(temp_list):
            policy_logs = policy_all_logps[temp_list]
            with torch.no_grad():
                ref_logs = ref_all_logps[temp_list]
            return policy_logs, ref_logs

        for idx1, idx2 in idx_pairs:
            o1 = object_values[idx1]
            o2 = object_values[idx2]
            
            if torch.all(o1>=o2) and torch.any(o1>o2):
                chosen_idx_list.append(idx1)
                rejected_idx_list.append(idx2)
            elif torch.all(o1<=o2) and torch.any(o1<o2):
                chosen_idx_list.append(idx2)
                rejected_idx_list.append(idx1)
            else:
                non_dominate_list[0].append(idx1)
                non_dominate_list[1].append(idx2)
        losses1 = torch.tensor([]).to(policy_all_logps.device)
        losses2 = torch.tensor([]).to(policy_all_logps.device)
        
        if chosen_idx_list:
            policy_chosen_logps, reference_chosen_logps = log_extract(chosen_idx_list)
            policy_rejected_logps, reference_rejected_logps = log_extract(rejected_idx_list)
            losses1, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, self.dpo_loss_config)
        
        
        if dpo_loss_config.nondominate_punishment is None:
            pass
        elif dpo_loss_config.nondominate_punishment == 'prob_diff':
            
            if non_dominate_list:
                policy_log1, ref_log1 = log_extract(non_dominate_list[0])
                policy_log2, ref_log2 = log_extract(non_dominate_list[1])
                epsilon = self.dpo_loss_config.epsilon  # Epsilon value
                loss_fn = EpsilonInsensitiveLoss(epsilon, reference_free=self.dpo_loss_config.reference_free)
                losses2 = loss_fn(policy_log1, policy_log2, ref_log1, ref_log2)
                
        losses = torch.concatenate((losses1, losses2))
        
        if len(losses) == 0:
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            loss = losses.mean()
        rewards_log['non-dominate_num'] = len(losses2)
        rewards_log['dominate_num'] = len(losses1)
        rewards_log['non-dominate_loss'] = losses2.mean() if len(losses2) else torch.tensor(0.0)
        return loss, rewards_log
    
    def sample_prompt_calculate_logp(self, batch, policy_model):
        (policy_logits, ref_logits, output_tokens, output_ids, sequence_lengths) = \
            policy_model._decode_sampling(batch=batch) # for multiobjective batch size should be large
        
        sample_num = policy_logits.shape[0]
        token_lenght = policy_logits.shape[1]
        
        policy_all_logps = policy_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        
        ref_all_logps = ref_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        
        sum_reward, multi_rewards_dict, tokens_list, rewards_log= \
            policy_model.compute_rewards(batch=batch, 
                                output_tokens=output_tokens,
                                mode="train", multi_optimize=self.dpo_loss_config.multi_optimize)
        
        return sum_reward, multi_rewards_dict, tokens_list, rewards_log, policy_all_logps, ref_all_logps

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
    # def pareto_front(self, reward1, reward2):
    #     combined_rewards = torch.stack([reward1, reward2], dim=1)
    
    #     # Sort the combined rewards in descending order
    #     sorted_rewards, indices = torch.sort(combined_rewards, dim=0, descending=True)
        
    #     # Initialize the Pareto front and other layers indices
    #     pareto_front_indices = [indices[0, 0]]
    #     other_layers_indices = []
        
    #     # Iterate through the sorted rewards to identify the Pareto front and other layers
    #     for i in range(1, len(sorted_rewards)):
    #         if sorted_rewards[i, 1] < sorted_rewards[i-1, 1]:
    #             pareto_front_indices.append(indices[i, 0])
    #         else:
    #             other_layers_indices.append(indices[i, 0])
        
    #     return pareto_front_indices, other_layers_indices
    
    # def find_dominate_indices(self, reward1, reward2):
    #     """
        # Find the dominated indices from two reward tensors and return a dictionary.
        
        # Parameters:
        #     reward1 (list or torch.Tensor): Reward tensor 1.
        #     reward2 (list or torch.Tensor): Reward tensor 2.
            
        # Returns:
        #     dict: A dictionary where keys are indices of dominating elements and values are lists 
        #         of indices dominated by the corresponding key.
        # """
        # dominate_dict = {}
        # for i, (r1, r2) in enumerate(zip(reward1, reward2)):
        #     for j, (rr1, rr2) in enumerate(zip(reward1, reward2)):
        #         if i != j:  # Avoid comparing an element with itself
        #             if r1 >= rr1 and r2 >= rr2 and (r1 > rr1 or r2 > rr2):
        #                 # Element i dominates element j
        #                 dominate_dict.setdefault(i, []).append(j)
        # return dominate_dict
        # self._target_model is the reference model
        #  outputs_ = self._target_model.teacher_forcing(**batch_) this one gives the logtis at: outputs_['sample_logits'].contiguous(),
            #    https://vscode.dev/github/mingkaid/rl-prompt/blob/main/rlprompt/modules/sql_module.py#L216
            
        # policy_chosen_logps = policy_logits[chosen_idx_list]
        # policy_rejected_logps = policy_logits[rejected_idx_list]
        # reference_chosen_logps = ref_logits[chosen_idx_list]
        # reference_rejected_logps = ref_logits[rejected_idx_list]
            
class EpsilonInsensitiveLoss(torch.nn.Module):
    def __init__(self, epsilon, reference_free=False):
        super(EpsilonInsensitiveLoss, self).__init__()
        self.epsilon = epsilon
        self.reference_free=reference_free

    def forward(self, policy_log1, policy_log2, ref_log1, ref_log2):
        if self.reference_free:
            logits = policy_log1-policy_log2
        else:
            logits = reward_diff(policy_log1, policy_log2, ref_log1, ref_log2)
        loss = torch.max(torch.abs(logits) - self.epsilon, torch.tensor(0.0))
        return loss


class DPO_O2_Trainer_test(DPO_O2_Trainer):
    def dpo_loss_calculation(self, step, batch, dpo_loss_config):

        

        policy_model = self.module.train()
        policy_model._pre_steps(step)  
        
        sum_reward, multi_rewards_dict, tokens_list, rewards_log, policy_all_logps, ref_all_logps = \
        self.sample_prompt_calculate_logp(batch, policy_model)
        
        # assert len(multi_rewards_dict.keys()) == 2, 'the key number must be 2'
        object_values = torch.stack(list(multi_rewards_dict.values()), axis=1)
        
        sample_num = len(sum_reward)
        idxs = list(range(sample_num))
        random.shuffle(idxs)
        
        dominance_list = [1]*sample_num # 1 means not dominated by the pareto front
        
        if self.pareto_front_dict:
            for i in range(sample_num):
                new_point = object_values[i]
                for key, tensor in self.pareto_front_dict.items():
                    if torch.all(tensor >= new_point):
                        dominance_list[i] = 0
                        break
        
        idx_pairs = [(idxs[i], idxs[i+1]) for i in range(0, sample_num, 2)]
        idx_pairs = [(i, j) for i in range(sample_num) for j in range(i+1, sample_num)]
        
        
        
        # if not self.pareto_front:
        #     loss = torch.tensor(0.0, requires_grad=True)
        #     return 0.0, 
        
        chosen_idx_list = []
        rejected_idx_list = []
        non_dominate_list = [[], []]

        def log_extract(temp_list):
            policy_logs = policy_all_logps[temp_list]
            with torch.no_grad():
                ref_logs = ref_all_logps[temp_list]
            return policy_logs, ref_logs

        for idx1, idx2 in idx_pairs:
            o1 = object_values[idx1]
            o2 = object_values[idx2]
            
            if torch.all(o1>=o2) and torch.any(o1>o2):
                chosen_idx_list.append(idx1)
                rejected_idx_list.append(idx2)
            elif torch.all(o1<=o2) and torch.any(o1<o2):
                chosen_idx_list.append(idx2)
                rejected_idx_list.append(idx1)
            elif dominance_list[idx1]==1 and dominance_list[idx2]==0:
                chosen_idx_list.append(idx1)
                rejected_idx_list.append(idx2)
            elif dominance_list[idx1]==0 and dominance_list[idx2]==1:
                chosen_idx_list.append(idx2)
                rejected_idx_list.append(idx1)
            else:
                non_dominate_list[0].append(idx1)
                non_dominate_list[1].append(idx2)
        losses1 = torch.tensor([]).to(policy_all_logps.device)
        losses2 = torch.tensor([]).to(policy_all_logps.device)
        
        if chosen_idx_list:
            policy_chosen_logps, reference_chosen_logps = log_extract(chosen_idx_list)
            policy_rejected_logps, reference_rejected_logps = log_extract(rejected_idx_list)
            losses1, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, self.dpo_loss_config)
        
        
        if dpo_loss_config.nondominate_punishment is None:
            pass
        elif dpo_loss_config.nondominate_punishment == 'prob_diff':
            
            if non_dominate_list:
                policy_log1, ref_log1 = log_extract(non_dominate_list[0])
                policy_log2, ref_log2 = log_extract(non_dominate_list[1])
                epsilon = self.dpo_loss_config.epsilon  # Epsilon value
                loss_fn = EpsilonInsensitiveLoss(epsilon, reference_free=self.dpo_loss_config.reference_free)
                losses2 = loss_fn(policy_log1, policy_log2, ref_log1, ref_log2)
                
        losses = torch.concatenate((losses1, losses2))
        
        if len(losses) == 0:
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            loss = losses.mean()
        rewards_log['non-dominate_num'] = len(losses2)
        rewards_log['dominate_num'] = len(losses1)
        rewards_log['non-dominate_loss'] = losses2.mean() if len(losses2) else torch.tensor(0.0)
        
        self.update_pareto_front(multi_rewards_dict, tokens_list)
        return loss, rewards_log
    
    
class DPO_O3_Trainer_test(DPO_O2_Trainer):
    def dpo_loss_calculation(self, step, batch, dpo_loss_config):

        

        policy_model = self.module.train()
        policy_model._pre_steps(step)  
        
        sum_reward, multi_rewards_dict, tokens_list, rewards_log, policy_all_logps, ref_all_logps = \
        self.sample_prompt_calculate_logp(batch, policy_model)
        
        # assert len(multi_rewards_dict.keys()) == 3, 'the key number must be 3'
        # object1_name, object2_name, object3_name = multi_rewards_dict.keys()
        # object1_value, object2_value, object3_value = multi_rewards_dict[object1_name], multi_rewards_dict[object2_name]
        
        object_values = torch.stack(list(multi_rewards_dict.values()), axis=1)
        
        sample_num = len(sum_reward)
        idxs = list(range(sample_num))
        random.shuffle(idxs)
        
        dominance_list = [1]*sample_num # 1 means not dominated by the pareto front
        
        if self.pareto_front_dict:
            for i in range(sample_num):
                
                new_point = object_values[i]
                for key, tensor in self.pareto_front_dict.items():
                    if torch.all(tensor >= new_point):
                        dominance_list[i] = 0
                        break
        
        idx_pairs = [(idxs[i], idxs[i+1]) for i in range(0, sample_num, 2)]
        idx_pairs = [(i, j) for i in range(sample_num) for j in range(i+1, sample_num)]
        
        
        
        # if not self.pareto_front:
        #     loss = torch.tensor(0.0, requires_grad=True)
        #     return 0.0, 
        
        chosen_idx_list = []
        rejected_idx_list = []
        non_dominate_list = [[], []]

        def log_extract(temp_list):
            policy_logs = policy_all_logps[temp_list]
            with torch.no_grad():
                ref_logs = ref_all_logps[temp_list]
            return policy_logs, ref_logs

        for idx1, idx2 in idx_pairs:
            o1 = object_values[idx1]
            o2 = object_values[idx2]
            
            if torch.all(o1>=o2) and torch.any(o1>o2):
                chosen_idx_list.append(idx1)
                rejected_idx_list.append(idx2)
            elif torch.all(o1<=o2) and torch.any(o1<o2):
                chosen_idx_list.append(idx2)
                rejected_idx_list.append(idx1)
            elif dominance_list[idx1]==1 and dominance_list[idx2]==0:
                chosen_idx_list.append(idx1)
                rejected_idx_list.append(idx2)
            elif dominance_list[idx1]==0 and dominance_list[idx2]==1:
                chosen_idx_list.append(idx2)
                rejected_idx_list.append(idx1)
            else:
                non_dominate_list[0].append(idx1)
                non_dominate_list[1].append(idx2)
        losses1 = torch.tensor([]).to(policy_all_logps.device)
        losses2 = torch.tensor([]).to(policy_all_logps.device)
        
        if chosen_idx_list:
            policy_chosen_logps, reference_chosen_logps = log_extract(chosen_idx_list)
            policy_rejected_logps, reference_rejected_logps = log_extract(rejected_idx_list)
            losses1, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, self.dpo_loss_config)
        
        
        if dpo_loss_config.nondominate_punishment is None:
            pass
        elif dpo_loss_config.nondominate_punishment == 'prob_diff':
            
            if non_dominate_list:
                policy_log1, ref_log1 = log_extract(non_dominate_list[0])
                policy_log2, ref_log2 = log_extract(non_dominate_list[1])
                epsilon = self.dpo_loss_config.epsilon  # Epsilon value
                loss_fn = EpsilonInsensitiveLoss(epsilon, reference_free=self.dpo_loss_config.reference_free)
                losses2 = loss_fn(policy_log1, policy_log2, ref_log1, ref_log2)
                
        losses = torch.concatenate((losses1, losses2))
        
        if len(losses) == 0:
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            loss = losses.mean()
        rewards_log['non-dominate_num'] = len(losses2)
        rewards_log['dominate_num'] = len(losses1)
        rewards_log['non-dominate_loss'] = losses2.mean() if len(losses2) else torch.tensor(0.0)
        
        self.update_pareto_front(multi_rewards_dict, tokens_list)
        return loss, rewards_log