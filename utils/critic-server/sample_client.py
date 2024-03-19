from openai import OpenAI
from llamagym import evaluate_state



question="You are a detective investigating a warehouse that has burned down, it may have been arson, find out the culprint"
agent_state= "I will investigate how many sprinkles are on the average donut"
goal_state="The owners did it for an insurance payout"

# an obvious sanity check

oai_resp = evaluate_state(agent_state,question ,goal_state)
print(oai_resp)

cpp_res = evaluate_state(agent_state,question ,goal_state, base_url="http://127.0.0.1:8000/v1")
print(cpp_res)