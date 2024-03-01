import os
import torch
from tqdm import trange
import gymnasium as gym
from transformers import AutoTokenizer
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from peft import LoraConfig
import wandb

HF_TOKEN = None

def llm(messages) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_ids = model.generate(
        inputs=inputs.input_ids,
        **{key.split('/')[-1]: value for key, value in hyperparams.items() if key.startswith('generate/')}
    )
    outputs = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    response = outputs[0].split("<|assistant|>\n")[-1]
    return response

system_prompt = """You are an expert single-player blackjack player. Every turn, you'll see your current sum, the dealer's showing card value, and whether you have a usable ace. You win by (1) not exceeding 21, and (2) exceeding the dealer's hand -- hidden+showing card values.
Decide whether to stick or hit by writing "Action: 0" or "Action: 1" respectively. You MUST decide your action and remember to write "Action:"."""

def format_observation(d) -> str:
    return f"Sum is {d[0]}, dealer is {d[1]}, {'have' if bool(d[2]) else 'no'} ace"  # # i.e. Sum is 14, dealer is 6, no ace.


def extract_action(response) -> int:
    digits = [char for char in response if char.isdigit()]
    if len(digits) == 0 or digits[-1] not in ("0", "1"):
        if "stick" in response.lower():
            return 0
        elif "hit" in response.lower():
            return 1
        raise ValueError("No action chosen")
    else:
        return int(digits[-1])


def batchify_episode(messages, reward):
    queries, responses = [], []
    for i in range(1, len(messages), 2):
        prompt = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        conversation_chunks = prompt.split("<|assistant|>\n")
        query = "<|assistant|>\n".join(conversation_chunks[:-1]) + "<|assistant|>\n"
        response = conversation_chunks[-1][:-5]  # remove final "</s>\n"

        query = tokenizer(query, return_tensors="pt").input_ids[0]
        response = tokenizer(response, return_tensors="pt").input_ids[0]

        queries.append(query)
        responses.append(response)

    per_turn_reward = reward / (
        len(messages) / 2
    )  # TODO: this is a hack. better credit assignment strats?
    rewards = [torch.tensor(per_turn_reward, dtype=torch.float16)] * len(queries)

    return queries, responses, rewards


if __name__ == "__main__":
    hyperparams = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "env": "Blackjack-v1",
        "lora/r": 16,
        "lora/lora_alpha": 32,
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": True,
        "batch_size": 16,
        "seed": 42069,
        "episodes": 5000,
        "generate/max_new_tokens": 32,
        "generate/do_sample": True,
        "generate/top_p": 0.6,
        "generate/top_k": 0,
        "generate/temperature": 0.9
    }
    wandb_run = wandb.init(project="LlamaGym", config=hyperparams)
    device = "cuda:0"

    lora_config = LoraConfig(
        **{key.split('/')[-1]: value for key, value in hyperparams.items() if key.startswith('lora/')}
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        peft_config=lora_config,
        load_in_8bit=hyperparams["load_in_8bit"],
        token=HF_TOKEN
    ).to(device)
    model_ref = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"], token=HF_TOKEN)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    model.pretrained_model.resize_token_embeddings(len(tokenizer))

    ppo_config = PPOConfig(batch_size=hyperparams['batch_size'])
    ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

    env = gym.make(hyperparams['env'], natural=False, sab=False)
    observation, info = env.reset(seed=hyperparams['seed'])
    trajectories = []
    batch = {"queries": [], "responses": [], "rewards": []}
    running_rewards = []
    running_mean_reward = 0

    for episode in trange(hyperparams['episodes']):
        episode_messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        terminated, truncated = False, False
        while True:
            message = format_observation(observation)
            episode_messages += [{"role": "user", "content": message}]

            response = llm(episode_messages)
            try:
                action = extract_action(response)
            except ValueError:
                break

            episode_messages += [{"role": "assistant", "content": response}]
            # print(f"Action: {action}")

            try:
                observation, reward, terminated, truncated, info = env.step(action)
                wandb.log({"action": action})
            except AssertionError:
                print("ERROR:", action)
                observation, info = env.reset()
                break

            if terminated or truncated:
                running_rewards.append(reward)
                if len(running_rewards) > 10:
                    running_rewards.pop(0)
                running_mean_reward = sum(running_rewards) / len(running_rewards)

                trajectories.append((episode_messages, observation, reward))
                wandb.log(
                    {
                        "episode": episode,
                        "reward": reward,
                        "reward_mean10": running_mean_reward,
                        "message_ct": len(episode_messages),
                    }
                )
                queries, responses, rewards = batchify_episode(episode_messages, reward)
                batch["queries"].extend(queries)
                batch["responses"].extend(responses)
                batch["rewards"].extend(rewards)

                # print(f"RESET {observation}, {reward}")
                observation, info = env.reset()
                break

        if len(batch["queries"]) >= ppo_config.batch_size:
            if len(batch["queries"]) > ppo_config.batch_size:
                queries = batch["queries"][ppo_config.batch_size :]
                batch["queries"] = batch["queries"][: ppo_config.batch_size]
                responses = batch["responses"][ppo_config.batch_size :]
                batch["responses"] = batch["responses"][: ppo_config.batch_size]
                rewards = batch["rewards"][ppo_config.batch_size :]
                batch["rewards"] = batch["rewards"][: ppo_config.batch_size]
            else:
                queries, responses, rewards = [], [], []

            train_stats = ppo_trainer.step(
                batch["queries"], batch["responses"], batch["rewards"]
            )
            wandb.log(train_stats)
            batch = {"queries": queries, "responses": responses, "rewards": rewards}

            torch.cuda.empty_cache()
        
        if episode % 1000 == 0:
            checkpoint_dir = f"./checkpoints/{wandb_run.name}/{str(episode)}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)

    env.close()

    model.save_pretrained(f"./checkpoints/{wandb_run.name}")