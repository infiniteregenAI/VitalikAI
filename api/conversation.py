from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import uvicorn
from api_vitalik_chat import VitalikAgent

# Define request model
class ChatRequest(BaseModel):
    user_input: str

# Initialize FastAPI app
app = FastAPI(title="Streaming VitalikAgent Chat API", description="Stream chat responses from VitalikAgent.")

@app.post("/stream_chat")
async def stream_chat(request: ChatRequest):
    """
    Stream chat responses from the VitalikAgent in real-time.

    Parameters:
    - user_input (str): User's query string.

    Returns:
    - Streaming response as plain text.
    """
    try:
        agent = VitalikAgent()  # Initialize the agent
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to initialize VitalikAgent.")

    async def event_generator():
        try:
            async for chunk in agent.stream_query(request.user_input):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
