
import os

import time 
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from fastapi.responses import StreamingResponse


class Prompt(BaseModel):
    prompt: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost.tiangolo.com",
        "https://localhost.tiangolo.com",
        "http://localhost",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def get_music(data: Prompt):
    print(f"data: {data}")
    file_path = "https://p.scdn.co/mp3-preview/f3992cadc24b4516addd63d779e9cde068e88bd7?cid=c5d155e14ff9463b8f34d250c31a2c36"
    
    time.sleep(2)
    return {"url": file_path}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)