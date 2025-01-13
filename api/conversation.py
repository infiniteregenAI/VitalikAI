from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from api.vitalik_chat import VitalikAgent

# Define request model
class ChatRequest(BaseModel):
    user_input: str

# Initialize FastAPI app
app = FastAPI(title="Streaming VitalikAgent Chat API", description="Stream chat responses from VitalikAgent.")

# Initialize the VitalikAgent instance
@app.on_event("startup")
async def initialize_agent():
    global agent
    try:
        agent = VitalikAgent()
    except Exception as e:
        print(f"Failed to initialize VitalikAgent: {str(e)}")
        raise RuntimeError("VitalikAgent initialization failed.")

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
        async def event_generator():
            try:
                # Stream response from the agent
                response = await agent.query(request.user_input)

                # Stream the response chunk by chunk
                for chunk in response["response"]:
                    yield chunk  # Send the chunk to the client
                    await asyncio.sleep(0)  # Yield control to the event loop
            except Exception as e:
                error_message = f"Streaming error: {str(e)}"
                yield error_message

        # Return the streaming response
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"}  # Disable buffering for real-time response
        )

    except Exception as e:
        print(f"Error during chat processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
