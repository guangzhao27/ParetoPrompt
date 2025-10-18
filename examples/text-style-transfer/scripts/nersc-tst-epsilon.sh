#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g

epsilon=$1


wandb login e48870a815eed7ccfabbb6d1a0e40f9a618bfa88
cd ./examples/text-style-transfer
python -u run_tst.py \
dataset=yelp \
report_to_wandb=true \
prompt_length=2 task_lm=gpt2-xl direction=0_to_1 \
dpo_loss_config.epsilon=$epsilon \
training_device='cuda' max_train_steps=10 project_name=iclr-task run_name=yelp-pl2-0to1topk200_$epsilon \
algorithm_name=ParetoPrompt-IPO \
eval_steps=5 dominate_evaluate_num=64 \
target_update_method="copy" target_update_steps=2 \
model_path=null load_step=0 \
top_k=200
# max_size=16
