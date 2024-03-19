
import threading
import uvicorn
import asyncio
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import os, json, datetime, re, time
from utils import printc, ChatCompletionRequest, to_dict
from llama_cpp import Llama, LlamaGrammar


from datetime import datetime

llm = Llama("./llama2-70b.gguf", n_gpu_layers=-1, n_ctx=16384)


streaming_semaphore = asyncio.Semaphore(1)


app = FastAPI()

password ="password"
def verify_api_key(authorization: str = Header(None)) -> None:
    expected_api_key = password
    if expected_api_key and (authorization is None or authorization != f"Bearer {expected_api_key}"):
        raise HTTPException(status_code=401, detail="Unauthorized")

# check_key = [Depends(verify_api_key)]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mutex lock
data_lock = threading.Lock()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI server!"}

@app.post('/v1/chat/completions')
async def openai_chat_completions(request: Request, request_data: ChatCompletionRequest):
    with data_lock:
        printc("Request Received","red")
        request_data= to_dict(request_data)
        printc(request_data,"yellow")
        msgs=request_data['messages']
        msgs[-1]['content']+=""" \n
        please respond in the following format:
            warmer_colder= whether the agent getting warmer/closer or colder/further away from the goal state. 
            answer warmer_colder by using one of the options: "hot", "warmer", "colder", "same", "cold"

            feedback= this is feedback to agent that should not give away the answer but should provivde constructive
            critism. Note that your are given the final goal state above so you should base your feedback on that. PROVIDE AT LEAST 5 WORDS OF FEEDBACK
        
            reached_goal= this either true or false if the agent has actually solved the problem and their state matches the goal state

        """
        res = llm.create_chat_completion(messages=msgs,
                                         temperature=request_data['temperature'],
                                         grammar=LlamaGrammar.from_file('reponse_format.gbnf'))

        output=json.loads(res['choices'][0]['message']["content"])
        output['feedback'] = output['feedback'].replace("\n", " ").replace('\r', '').replace("<|im_end|>", "").replace("<|im_start|>","").replace("assistant","").strip()
        res['choices'][0]['message']["content"]=None
        res['model']=request_data['model']
        res['choices'][0]['message']["function_call"]={"name":request_data["functions"][0]['name'],"arguments":json.dumps(output) }
        printc(res['choices'][0]['message']["function_call"],"green")

        return res



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
