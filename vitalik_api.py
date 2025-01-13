from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from vitalik_chat import VitalikAgent

# Define request model
class ChatRequest(BaseModel):
    user_input: str

# Initialize FastAPI app
app = FastAPI(title="VitalikAgent Chat API", description="A chat endpoint for VitalikAgent.")

# Initialize the VitalikAgent instance
@app.on_event("startup")
async def initialize_agent():
    global agent
    try:
        agent = VitalikAgent()
    except Exception as e:
        print(f"Failed to initialize VitalikAgent: {str(e)}")
        raise RuntimeError("VitalikAgent initialization failed.")

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint to interact with VitalikAgent.

    Parameters:
    - user_input (str): User's query string.

    Returns:
    - The agent's reply as plain text.
    """
    try:
        result = await agent.query(request.user_input)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result["response"]
    except Exception as e:
        print(f"Error during chat processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
