import json
import asyncio
from string import Formatter
from typing import List, Dict, Any
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
import chromadb
from langchain.callbacks.base import BaseCallbackHandler
import chainlit as cl
from models import ReasoningLayer
from utils import VitalikUtils
from vitalik_agent import VitalikAgent

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, msg: cl.Message):
        super().__init__()
        self.msg = msg
        self.sources = set()
        self.response_chunks = []

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.msg.stream_token(token)
        self.response_chunks.append(token)

    async def on_llm_end(self, response, **kwargs):
        if self.sources:
            self.msg.elements.append(
                cl.Text(
                    name="Referenced Sources", 
                    content="From my writings and research:\n" + "\n".join(sorted(self.sources)),
                    display="inline"
                )
            )

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        if "search" in serialized.get("name", ""):
            source_type = serialized["name"].replace("search_", "").replace("_knowledge", "")
            self.sources.add(f"Consulting {source_type} knowledge base")

def format_context(context_results):
    if not context_results:
        return "No specific historical context found for this topic."
    return "\n".join(f"- {result.get('content', str(result))}" for result in context_results)

@cl.on_chat_start
async def on_chat_start():
    """Handles the initialization of the VitalikAgent."""
    try:
        print("Initializing VitalikAgent...")
        agent = VitalikAgent()  # Initialize the VitalikAgent
        cl.user_session.set("agent", agent)
        print("Agent initialized successfully!")
        # Do not send this as output to the UI
    except Exception as e:
        error_msg = f"Error initializing agent: {str(e)}"
        print(error_msg)
        # Send an error message to the user only if something goes wrong
        await cl.Message(content="An error occurred during initialization. Please try restarting the chat.", author="system").send()

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Ethereum Scaling",
            message="What are your thoughts on the current state of Ethereum scaling solutions, particularly Layer 2 solutions and their impact on the ecosystem?",
            icon="/public/logo_dark.png",
        ),
        cl.Starter(
            label="Proof of Stake vs Proof of Work",
            message="Can you explain the key differences between Proof of Stake and Proof of Work consensus mechanisms, and why Ethereum transitioned to PoS?",
            icon="/public/logo_dark.png",
        ),
        cl.Starter(
            label="Future of Blockchain",
            message="What's your vision for the future of blockchain technology and its potential impact on society beyond just financial applications?",
            icon="/public/logo_dark.png",
        ),
        cl.Starter(
            label="Cryptoeconomics",
            message="Could you explain the fundamental principles of cryptoeconomics and how they apply to blockchain system design?",
            icon="/public/logo_dark.png",
        )
    ]

@cl.on_message
async def on_message(message: cl.Message):
    """Processes incoming messages and routes them through the VitalikAgent."""
    agent = cl.user_session.get("agent")
    
    if not agent:
        # Ensure the agent is initialized before proceeding
        await cl.Message(content="Agent not initialized. Please restart the chat.", author="vitalikAI").send()
        return

    async with cl.Step(name="Process Message") as step:
        msg = cl.Message(content="", author="vitalikAI")
        try:
            # Determine message type
            classification_prompt = (
                """Determine if this is a conversational message that doesn't require technical knowledge:
                Message: {message}

                If this is a simple greeting, personal question, a general question like 'who made you' or general conversation
                that doesn't require technical knowledge, return CONVERSATIONAL. Otherwise, return NEEDS_KNOWLEDGE.

                Just return one word: CONVERSATIONAL or NEEDS_KNOWLEDGE."""
            )

            classification_response = await agent.llm.ainvoke(
                [{"role": "user", "content": classification_prompt.format(message=message.content)}]
            )
            is_conversational = "CONVERSATIONAL" in classification_response.content.upper()

            if is_conversational:
                # Handle conversational messages
                conversation_prompt = (
                    """As {name}, respond naturally to this message while maintaining my characteristic analytical style and warmth:

                    Message: {message}"""
                )
                stream_handler = StreamingCallbackHandler(msg)
                await agent.agent.ainvoke(
                    {"input": conversation_prompt.format(name=agent.persona['name'], message=message.content)},
                    config={"callbacks": [stream_handler]}
                )
            else:
                # Handle knowledge-intensive messages
                technical_context = await agent._search_technical_db(message.content)
                blog_context = await agent._search_blog_db(message.content)
                temporal_context = await agent._search_temporal_db(message.content)

                technical_prompt = (
                    """You are {name}, {role}. You have been asked: {message}

                    Consider this context from your knowledge base:

                    Technical Research:
                    {technical}

                    Past Writings:
                    {blog}

                    Historical Perspective:
                    {temporal}

                    Important: Respond naturally as Vitalik Buterin. While your analysis should be grounded in first principles and
                    consider technical, economic, and social implications, avoid explicitly stating your thought process or using headers.

                    Focus on providing clear insights while naturally incorporating:
                    - Technical accuracy and mathematical precision when relevant
                    - References to research and past writings where applicable
                    - Balanced consideration of counterarguments
                    - Long-term implications"""
                )

                stream_handler = StreamingCallbackHandler(msg)
                await agent.agent.ainvoke(
                    {
                        "input": technical_prompt.format(
                            name=agent.persona['name'],
                            role=agent.persona['role'],
                            message=message.content,
                            technical=format_context(technical_context),
                            blog=format_context(blog_context),
                            temporal=format_context(temporal_context)
                        )
                    },
                    config={"callbacks": [stream_handler]}
                )

            step.output = "Message processed successfully."
            await msg.send()
        except Exception as e:
            step.output = f"Error processing message: {str(e)}"
            await cl.Message(content=step.output, author="Agent Vitalik").send()
