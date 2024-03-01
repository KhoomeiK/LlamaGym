import torch
from transformers import AutoTokenizer
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from trl.core import respond_to_batch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cpu"

# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
model_ref = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# initialize trainer
ppo_config = PPOConfig(
    batch_size=1,
)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(device)

# get model response
response_tensor = respond_to_batch(model, query_tensor)

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
# NOTE: currently optimizer.step() is commented out
print(train_stats)