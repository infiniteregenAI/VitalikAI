from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from vitalik import VitalikAgent

# Define request and response models
class QueryRequest(BaseModel):
    user_input: str

class QueryResponse(BaseModel):
    response: str
    context: dict

# Initialize FastAPI app
app = FastAPI(title="VitalikAgent API", description="API for interacting with VitalikAgent.")

# Initialize the VitalikAgent instance
@app.on_event("startup")
async def initialize_agent():
    global agent
    try:
        agent = VitalikAgent()
    except Exception as e:
        print(f"Failed to initialize VitalikAgent: {str(e)}")
        raise RuntimeError("VitalikAgent initialization failed.")

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Endpoint to query the VitalikAgent.

    Parameters:
    - user_input (str): User's query string.

    Returns:
    - JSON response with agent's output and context.
    """
    try:
        result = await agent.query(request.user_input)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return QueryResponse(
            response=result["response"],
            context=result["context"]
        )
    except Exception as e:
        print(f"Error during query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
