

from openai import OpenAI
import json


def evaluate_state(agent_state:str,question:str, goal_state:str, model="gpt-3.5-turbo-0125", base_url:str=None):
    client = OpenAI() if base_url is None else OpenAI(base_url=base_url)
    score = 0
    response = client.chat.completions.create(
        model=model,
        max_tokens=120,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that helps evuluate an agent at problem solving. Given a question and a goal state, evaluate the agent's current state",
            },
            {
                "role": "user",
                "content": str(
                    "QUESTION/START STATE:"
                    + question
                    + "\nAGENT CURRENT STATE: "
                    + agent_state
                    + "\n:CORRECT GOAL STATE: "
                    + goal_state
                ),
            },
        ],
        functions=[
            {
                "name": "evaluate_state",
                "description": "Evaluates the agent's answer compared to the answer key",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "warmer_colder": {
                            "type": "string",
                            "enum":[ "hot", "warmer", "colder", "same", "cold"],
                            "description": "is the agent's current state closer or further from the goal state",
                        },
                        "feedback": {
                            "type": "string",
                            "description": "in a few short words, provide feedback to the agent on their answer",
                        },
                        "reached_goal": {
                            "type": "boolean",
                            "description": "true if the agent's current state the goal state, otherwise false",
                        },

                    },
                },
            }
        ],
    )

    
    data = json.loads(response.choices[0].message.function_call.arguments)
    
    #these are just sample metrics
    score += 1000 if data.get("reached_goal", False) else 0
    score += 100 if data.get("warmer_colder") == "hot" and not data.get("reached_goal", False) else 0
    score += 50 if data.get("warmer_colder") == "warmer" and not data.get("reached_goal", False) else 0
    # No need for a line for "same" as it does not change the score
    score -= 50 if data.get("warmer_colder") == "colder" and not data.get("reached_goal", False) else 0
    score -= 100 if data.get("warmer_colder") == "cold" and not data.get("reached_goal", False) else 0
    score -= 200 if data.get("is_nonsense", False) else 0

    feedback = data.get("feedback","No feedback")

    return {'reward': score, 'feedback': feedback, 'done':data.get("reached_goal", False)}



