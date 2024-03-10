<p align="center">
  <img src="https://raw.githubusercontent.com/khoomeik/LlamaGym/main/llamagym.png" height="250" alt="Llama Gym" />
</p>
<p align="center">
  <em>Fine-tune LLM agents with online reinforcement learning</em>
</p>
<p align="center">
    <a href="https://pypi.org/project/llamagym/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/llamagym?style=for-the-badge&color=3670A0">
    </a>
</p>
<p align="center">
<a href="https://reworkd.ai/">🔗 Agents for Web Data Extraction</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://twitter.com/khoomeik">🐦 Twitter</a>


# LlamaGym
"Agents" originated in reinforcement learning, where they learn by interacting with an environment and receiving a reward signal. However, LLM-based agents today do not learn online (i.e. continuously in real time) via reinforcement.

OpenAI created [Gym](https://github.com/Farama-Foundation/Gymnasium) to standardize and simplify RL environments, but if you try dropping an LLM-based agent into a Gym environment for training, you'd find it's still quite a bit of code to handle LLM conversation context, episode batches, reward assignment, PPO setup, and more.

LlamaGym seeks to simplify fine-tuning LLM agents with RL. Right now, it's a single `Agent` abstract class that handles all the issues mentioned above, letting you quickly iterate and experiment with agent prompting & hyperparameters across any Gym environment.

## Usage
Fine-tuning an LLM-based agent to play in a Gym-style environment with RL has never been easier!

First, implement 3 abstract methods on the Agent class:
```python
from llamagym import Agent

class BlackjackAgent(Agent):
    def get_system_prompt(self) -> str:
        return "You are an expert blackjack player."

    def format_observation(self, observation) -> str:
        return f"Your current total is {observation[0]}"

    def extract_action(self, response: str):
        return 0 if "stick" in response else 1
```

Then, define your base LLM (as you would for any fine-tuning job) and instantiate your agent:
```python
model = AutoModelForCausalLMWithValueHead.from_pretrained("Llama-2-7b").to(device)
tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b")
agent = BlackjackAgent(model, tokenizer, device)
```

Finally, write your RL loop as usual and simply call your agent to act, reward, and terminate:
```python
env = gym.make("Blackjack-v1")

for episode in trange(5000):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.act(observation) # act based on observation
        observation, reward, terminated, truncated, info = env.step(action)
        agent.assign_reward(reward) # provide reward to agent
        done = terminated or truncated

    train_stats = agent.terminate_episode() # trains if batch is full
```

Note: the above code snippets are mildly simplified but a fully working example is available in `examples/blackjack.py`.

## Relevant Work
- [Grounding Large Language Models with Online Reinforcement Learning](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)
  - [Lamorel: Language Models for Reinforcement Learning](https://github.com/flowersteam/lamorel)
- [True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning](https://github.com/WeihaoTan/TWOSOME)

## TODO
- [ ] add toggle for train vs inference mode
- [ ] set up logging on examples
  - [ ] approx KL, sample completions, sliding window mean reward
- [ ] understand the PPO logs and fix hyperparams
- [ ] run wandb hyperparam sweep
