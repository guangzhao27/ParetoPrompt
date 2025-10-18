import torch
from torch import optim, nn
import numpy as np
import random
from typing import Callable, List, Optional

def get_default_train_op(model: nn.Module,
                         optimizer: optim.Optimizer,
                         gradient_clip: bool,
                         gradient_clip_norm: float) -> Callable[[], None]: 
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=learning_rate)

    def _train_op(): 
        if gradient_clip: 
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        return optimizer.state_dict()

    return _train_op

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_pareto_front(samples):
    """
    Find the Pareto front of given sample points.
    
    :param samples: Torch tensor of size N*2 containing sample points (N samples, 2 objectives).
    :return: Pareto front as a torch tensor.
    """
    pareto_front = []
    dominated_by = torch.zeros(samples.size(0), dtype=torch.bool)
    
    for idx, point1 in enumerate(samples):
        if not dominated_by[idx]:
            pareto_front.append(idx)
            for j, point2 in enumerate(samples):
                if idx != j:
                    if torch.all(point1 >= point2):
                        dominated_by[j] = True
                    elif torch.all(point1 <= point2):
                        dominated_by[idx] = True
                        pareto_front.remove(idx)
                        break
    pareto_front_tensor = samples[pareto_front]
    
    return pareto_front_tensor

def calculate_dominating_volume(pareto_front:torch.Tensor, ref_point):
    """
    Calculate the dominating volume of the Pareto front with respect to a reference point.
    
    all objectives are maximizing

    """
    
    import pygmo as pg

    # Example Pareto front: A list of 3D points (x, y, z)
    pareto_front = -pareto_front
    ref_point = -ref_point
    
    

    # Reference point (should be worse than any point in the Pareto front)
    # reference_point = [.0, .0, .0]

    # Create a hypervolume object
    hv = pg.hypervolume(pareto_front)

    # Compute the hypervolume w.r.t. the reference point
    hypervolume_value = hv.compute(ref_point)
    return hypervolume_value
    
    
def calculate_dominating_volume2(pareto_front:torch.Tensor, ref_point):
    # Sort Pareto front based on the first objective (ascending order)
    sorted_pareto_front = pareto_front[pareto_front[:, 0].argsort()]
    
    # Initialize dominating volume
    dominating_volume = 0.0
    
    # Initialize the upper left corner of the rectangle
    upper_left_corner = ref_point.clone()
    
    # Iterate through sorted Pareto front
    for point in sorted_pareto_front:
        # Calculate the width and height of the rectangle
        width = point[0] - upper_left_corner[0]  
        height = point[1] - upper_left_corner[1]  
        
        # Update dominating volume by adding the area of the rectangle
        dominating_volume += width * height
        
        # Update the upper left corner for the next rectangle
        upper_left_corner[0] = point[0]
    
    return dominating_volume.item()


def evaluate_model_dominate_volume(model, batch, reference_point=torch.tensor([0., 0.]), dominate_evaluate_num=16, output_tokens_P:Optional[List]=None, different_reward=None):
    # dominate_evaluate_num: the num of prompts generated to evaluate the Pareto Front
    (logits, logits_, output_tokens, output_ids, sequence_lengths) = model._decode_sampling(batch=batch, batch_size=dominate_evaluate_num)
    if output_tokens_P is not None:
        output_tokens = output_tokens_P+output_tokens
    object1_list = []
    object2_list = []
    objects_value_list = []
    objects_name_list = []
    for prompt in output_tokens:    
        output_token_list = [prompt]*len(batch['source_texts'])
        if different_reward:
            sum_reward, multi_rewards_dict, tokens_list, rewards_log = different_reward(**batch, output_tokens=output_token_list, 
                                                                                        to_tensor=True, mode='infer',  multi_optimize=True)
        else:
            sum_reward, multi_rewards_dict, tokens_list, rewards_log= model.compute_rewards(batch=batch, output_tokens=output_token_list, multi_optimize=True)
        # if not objects_value_list:
        #     objects_value_list = [[]]*len(multi_rewards_dict)
        object_tensor = torch.stack(list(multi_rewards_dict.values()), axis=1)
        objects_value_list.append(object_tensor.mean(axis=0))
        # object1_name, object2_name = multi_rewards_dict.keys()
        # object1_value, object2_value = multi_rewards_dict[object1_name], multi_rewards_dict[object2_name]
        # object1_list.append(object1_value.mean().item())
        # object2_list.append(object2_value.mean().item())
    objects_name_list = list(multi_rewards_dict.keys())
    # object1 = torch.tensor(object1_list)
    # object2 = torch.tensor(object2_list)
    # samples = torch.stack((object1, object2), dim=1)
    samples = torch.stack(objects_value_list)
    pareto_front_tensor = find_pareto_front(samples)
    pareto_size = len(pareto_front_tensor)
    reference_point = torch.tensor([0.]*len(multi_rewards_dict))
    dominating_volume = calculate_dominating_volume(pareto_front_tensor, reference_point)
    
    performance = {}
    performance['dominating_volume'] = dominating_volume
    # performance[object1_name] = object1.tolist()
    # performance[object2_name] = object2.tolist()
    performance['pareto_size'] = pareto_size
    performance['tokens'] = output_tokens
    for i, key in enumerate(objects_name_list):
        performance[key] = samples[:, i].tolist()
    
    return dominating_volume, performance