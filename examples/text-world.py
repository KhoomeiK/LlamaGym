import os
from tqdm import trange
import wandb
import textworld.gym
from llamagym import Agent

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead


class TextworldAgent(Agent):
    def get_system_prompt(self) -> str:
        return "You will be playing a text-based game. Here are some example commands: 'go west', 'inventory', 'drop teacup', 'examine broom', 'close type 2 box', 'open door', 'insert pencil into type 1 box', 'look'. Not all commands will work, but there are many others that you can try beyond these examples. When responding, first reason about the game state to decide the best action and then say 'command: <your command>'."

    def format_observation(self, observation) -> str:
        # remove the game header text
        observation = observation.split("$$$$$$$ \n\n")[-1].strip()
        return observation

    def extract_action(self, response: str):
        if "command: " not in response:
            raise ValueError("No action chosen")
        command = response.split("command: ")[-1]
        return command


if __name__ == "__main__":
    hyperparams = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "lora/r": 16,
        "lora/lora_alpha": 32,
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": True,
        "batch_size": 16,
        "seed": 42069,
        "episodes": 1000,
        "generate/max_new_tokens": 32,
        "generate/do_sample": True,
        "generate/top_p": 0.6,
        "generate/top_k": 0,
        "generate/temperature": 0.9,
    }
    wandb_run = wandb.init(project="LlamaGym", config=hyperparams)
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

    agent = TextworldAgent(
        model,
        tokenizer,
        device,
        {
            key: value
            for key, value in hyperparams.items()
            if key.startswith("generate/")
        },
        {
            "batch_size": hyperparams["batch_size"]
        },  # TODO: figure out minibatch size issue
    )

    env_id = textworld.gym.register_game(
        "examples/tw_games/custom_game.z8",
        max_episode_steps=50,
        request_infos=textworld.EnvInfos(
            admissible_commands=True,
        ),
    )
    env = textworld.gym.make(env_id)

    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        env.render()
        done = False

        while not done:
            action = agent.act(observation)
            wandb.log({"action": action})
            observation, reward, done, info = env.step(action)
            env.render()
            agent.assign_reward(reward)

        episode_stats = {
            "episode": episode,
            "total_return": sum(agent.current_episode_rewards),
            "message_ct": len(agent.current_episode_messages),
            "episode_messages": agent.current_episode_messages[-1],
        }
        episode_stats.update(agent.terminate_episode())  # add train stats
        wandb.log(episode_stats)

    env.close()
