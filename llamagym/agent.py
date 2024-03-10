from abc import ABC, abstractmethod
from typing import List, Dict

import gymnasium as gym
import torch
from trl import (
    PPOTrainer,
    PPOConfig,
    create_reference_model,
)


class Agent(ABC):
    def __init__(
        self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None
    ):
        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_p": 0.6,
                "top_k": 0,
                "temperature": 0.9,
            }
        if ppo_config_dict is None:
            ppo_config_dict = {"batch_size": 16, "mini_batch_size": 16}

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_config_dict = generate_config_dict
        self.model_ref = create_reference_model(model)
        self.ppo_config = PPOConfig(**ppo_config_dict)
        self.ppo_trainer = PPOTrainer(self.ppo_config, model, self.model_ref, tokenizer)

        self.current_batch = {"queries": [], "responses": [], "rewards": []}

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_episode_rewards = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def format_observation(self, observation: gym.core.ObsType) -> str:
        pass

    @abstractmethod
    def extract_action(self, response: str) -> gym.core.ActType:
        pass

    def llm(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            inputs=inputs.input_ids,
            **{
                key.split("/")[-1]: value
                for key, value in self.generate_config_dict.items()
            }
        )
        outputs = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = outputs[0].split("[/INST]")[-1].strip()

        return response

    def act(self, observation):
        message = self.format_observation(observation)
        self.current_episode_messages += [{"role": "user", "content": message}]

        response = self.llm(self.current_episode_messages)
        try:
            action = self.extract_action(response)
        except Exception as e:
            return None

        self.current_episode_messages += [{"role": "assistant", "content": response}]
        return action

    def assign_reward(self, reward):
        self.current_episode_rewards.append(reward)

    def format_episode_for_ppo(self, messages, rewards):
        queries, responses = [], []
        for i in range(2, len(messages), 2):
            prompt = self.tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False
            )
            conversation_chunks = prompt.split("[/INST] ")
            query = '[/INST] '.join(conversation_chunks[:-1]) + "[/INST] "
            response = conversation_chunks[-1]

            query = self.tokenizer(query, return_tensors="pt").input_ids[0]
            response = self.tokenizer(response, return_tensors="pt").input_ids[0]

            queries.append(query)
            responses.append(response)

        if all(reward == 0 for reward in rewards[:-1]):
            # if sparse rewards, give equal reward to all conversation turns
            per_turn_reward = rewards[-1] / (len(messages) / 2)
            rewards = [torch.tensor(per_turn_reward, dtype=torch.float16)] * len(
                queries
            )
        else:
            rewards = [torch.tensor(reward, dtype=torch.float16) for reward in rewards]

        return queries, responses, rewards

    def terminate_episode(self, train=True):
        if train:
            queries, responses, rewards = self.format_episode_for_ppo(
                self.current_episode_messages, self.current_episode_rewards
            )

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_episode_rewards = []

        if train:
            self.current_batch["queries"].extend(queries)
            self.current_batch["responses"].extend(responses)
            self.current_batch["rewards"].extend(rewards)
    
            if len(self.current_batch["queries"]) >= self.ppo_config.batch_size:
                train_stats = self.train_batch(
                    self.current_batch["queries"],
                    self.current_batch["responses"],
                    self.current_batch["rewards"],
                )
                return train_stats

        return {}

    def train_batch(self, batch_queries, batch_responses, batch_rewards):
        if len(batch_queries) > self.ppo_config.batch_size:
            queries = batch_queries[: self.ppo_config.batch_size]
            responses = batch_responses[: self.ppo_config.batch_size]
            rewards = batch_rewards[: self.ppo_config.batch_size]

            # keep the remainder for the next batch
            self.current_batch["queries"] = batch_queries[self.ppo_config.batch_size :]
            self.current_batch["responses"] = batch_responses[
                self.ppo_config.batch_size :
            ]
            self.current_batch["rewards"] = batch_rewards[self.ppo_config.batch_size :]
        else:
            queries, responses, rewards = batch_queries, batch_responses, batch_rewards
            self.current_batch = {"queries": [], "responses": [], "rewards": []}

        train_stats = self.ppo_trainer.step(queries, responses, rewards)
        torch.cuda.empty_cache()

        return train_stats
