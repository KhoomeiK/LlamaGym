import os
from tqdm import trange

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

import gymnasium as gym
from llamagym import Agent


class BlackjackAgent(Agent):
    def get_system_prompt(self) -> str:
        return """You are an expert single-player blackjack player. Every turn, you'll see your current sum, the dealer's showing card value, and whether you have a usable ace. You win by (1) not exceeding 21, and (2) exceeding the dealer's hand -- hidden+showing card values.
Decide whether to stick or hit by writing "Action: 0" or "Action: 1" respectively. You MUST decide your action and remember to write "Action:"."""

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"Sum is {observation[0]}, dealer is {observation[1]}, {'have' if bool(observation[2]) else 'no'} ace"  # # i.e. Sum is 14, dealer is 6, no ace.

    def extract_action(self, response: str) -> gym.core.ActType:
        digits = [char for char in response if char.isdigit()]
        if len(digits) == 0 or digits[-1] not in ("0", "1"):
            if "stick" in response.lower():
                return 0
            elif "hit" in response.lower():
                return 1
            raise ValueError("No action chosen")
        else:
            return int(digits[-1])


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
        "generate/temperature": 0.9,
    }
    device = "cuda:0"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    lora_config = LoraConfig(
        **{
            key.split("/")[-1]: value
            for key, value in hyperparams.items()
            if key.startswith("lora/")
        }
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        peft_config=lora_config,
        load_in_8bit=hyperparams["load_in_8bit"],
        token=HF_TOKEN,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"], token=HF_TOKEN)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.pretrained_model.resize_token_embeddings(len(tokenizer))

    agent = BlackjackAgent(
        model,
        tokenizer,
        device,
        {
            key: value
            for key, value in hyperparams.items()
            if key.startswith("generate/")
        },
        {"batch_size": hyperparams["batch_size"]},
    )
    env = gym.make(hyperparams["env"], natural=False, sab=False)

    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset(seed=hyperparams["seed"])
        done = False

        while not done:
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            agent.assign_reward(reward)
            done = terminated or truncated

        train_stats = agent.terminate_episode()
