from transformers import AutoTokenizer, pipeline



task_lm = 'distilgpt2'
template = '{prompt} "{sentence_1}" "'
end_punct = '"'
pad_token = '<|endoftext|>'
generator_device = 'cuda'
lower_outputs = True
control_output_length = True

tokenizer = AutoTokenizer.from_pretrained(task_lm, pad_token=pad_token, save_directory="../llm_cache_dir")
print("2")
generator = pipeline("text-generation",
                     model=task_lm,
                                  tokenizer=tokenizer,
                                  device='cuda')
print("3")