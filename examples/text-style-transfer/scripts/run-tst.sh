cd ./examples/text-style-transfer
python -u run_tst.py \
dataset=yelp \
report_to_wandb=true \
prompt_length=5 task_lm=gpt2-xl direction=0_to_1 \
dpo_loss_config.epsilon=0.1 \
training_device='cuda' max_train_steps=10000 project_name=iclr-task run_name=yelp-pl2-0to1topk200_0.1 \
algorithm_name=ParetoPrompt-IPO \
eval_steps=500 dominate_evaluate_num=64 \
target_update_method="copy" target_update_steps=200 \
model_path=null load_step=0 \
top_k=200
# max_size=16
