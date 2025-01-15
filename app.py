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
        """Add relevant sources and context"""
        if self.sources:
            self.msg.elements.append(
                cl.Text(
                    name="Referenced Sources", 
                    content="From my writings and research:\n" + "\n".join(sorted(self.sources)),
                    display="inline"
                )
            )

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Track which knowledge bases are being accessed"""
        if "search" in serialized.get("name", ""):
            source_type = serialized["name"].replace("search_", "").replace("_knowledge", "")
            self.sources.add(f"Consulting {source_type} knowledge base")

def format_context(context_results):
    """Format context results for prompt input"""
    if not context_results:
        return "No specific historical context found for this topic."
        
    formatted = []
    for result in context_results:
        if isinstance(result, dict):
            formatted.append(f"- {result.get('content', '')}")
        else:
            formatted.append(f"- {str(result)}")
            
    return "\n".join(formatted)


class VitalikAgent:
    def _format_nested(self, template: str, data_dict: Dict) -> str:
        """Format string with nested dictionary access"""
        try:
            def resolve_key(key_path):
                parts = key_path.split('.')
                value = data_dict
                for part in parts:
                    if '[' in part:
                        array_name, index = part.split('[')
                        index = int(index.rstrip(']'))
                        value = value[array_name][index]
                    else:
                        value = value[part]
                return str(value)

            result = template
            while '{' in result and '}' in result:
                start = result.find('{')
                end = result.find('}')
                if start == -1 or end == -1:
                    break
                
                key = result[start+1:end]
                try:
                    value = resolve_key(key)
                    result = result[:start] + value + result[end+1:]
                except (KeyError, IndexError) as e:
                    print(f"Error resolving key {key}: {str(e)}")
                    result = result[:start] + f"[Error: {key} not found]" + result[end+1:]
            
            return result

        except Exception as e:
            print(f"Error in _format_nested: {str(e)}")
            print(f"Template: {template}")
            print(f"Data dict: {data_dict}")
            return str(e)

    def __init__(self, persona_path: str = 'persona.json'):
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for persona file at: {os.path.abspath(persona_path)}")
        
        # Load persona configuration
        try:
            with open(persona_path, 'r', encoding='utf-8') as f:
                self.persona = json.load(f)
                print(f"Successfully loaded persona data: {self.persona}")
                
                required_fields = ['name', 'role', 'system_prompt_template']
                missing_fields = [field for field in required_fields if field not in self.persona]
                if missing_fields:
                    raise ValueError(f"Missing required fields in persona.json: {missing_fields}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Persona file not found at {persona_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {persona_path}. Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading persona file: {str(e)}")
        
        # Initialize LLM with streaming
        try:
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                streaming=True
            )
        except Exception as e:
            print(f"Error initializing ChatOpenAI: {str(e)}")
            raise

        # Initialize Utils
        self.utils = VitalikUtils(self.llm)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize vector DB clients
        try:
            self.tech_db = chromadb.PersistentClient(path="./vectordbs/technical")
            self.blog_db = chromadb.PersistentClient(path="./vectordbs/blog")
            self.temporal_db = chromadb.PersistentClient(path="./vectordbs/temporal")
            
            self.tech_collection = self.tech_db.get_collection("technical_knowledge")
            self.blog_collection = self.blog_db.get_collection("blog_knowledge")
            self.temporal_collection = self.temporal_db.get_collection("temporal_knowledge")
        except Exception as e:
            print(f"Error initializing vector databases: {str(e)}")
            raise
        
        # Initialize tools and agent
        try:
            self.tools = self._create_tools()
            self.agent = self._create_agent()
            print("Successfully initialized agent!")
        except Exception as e:
            print(f"Error creating tools or agent: {str(e)}")
            raise

    async def _search_technical_db(self, query: str, n_results: int = 3) -> List[Dict]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.tech_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return self.utils.format_results(results, "technical")
        except Exception as e:
            print(f"Error in technical search: {str(e)}")
            return []

    async def _search_blog_db(self, query: str, n_results: int = 3) -> List[Dict]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.blog_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return self.utils.format_results(results, "blog")
        except Exception as e:
            print(f"Error in blog search: {str(e)}")
            return []

    async def _search_temporal_db(self, query: str, n_results: int = 3) -> List[Dict]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.temporal_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return self.utils.format_results(results, "temporal")
        except Exception as e:
            print(f"Error in temporal search: {str(e)}")
            return []

    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search_technical_knowledge",
                description="Search through technical papers and documentation about Ethereum",
                func=self._search_technical_db,
                coroutine=self._search_technical_db
            ),
            Tool(
                name="search_blog_knowledge",
                description="Search through my blog posts and articles",
                func=self._search_blog_db,
                coroutine=self._search_blog_db
            ),
            Tool(
                name="search_temporal_knowledge",
                description="Search through my tweets and historical content",
                func=self._search_temporal_db,
                coroutine=self._search_temporal_db
            ),
            Tool(
                name="analyze_layers",
                description="Analyze a topic across multiple dimensions",
                func=self._layered_analysis,
                coroutine=self._layered_analysis
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        try:
            simple_prompt = f"""I am {self.persona['name']}, {self.persona['role']}. 
            I approach problems from first principles, combining technical depth with philosophical considerations."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", simple_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
        except Exception as e:
            print(f"Error in _create_agent: {str(e)}")
            raise

    async def _layered_analysis(self, topic: str) -> List[ReasoningLayer]:
        layers = []
        
        try:
            # Layer 1: Technical Understanding
            tech_results = await self._search_technical_db(topic)
            layers.append(ReasoningLayer(
                name="Technical Analysis",
                description="Understanding the technical fundamentals",
                thought_process=self.utils.synthesize_thoughts(tech_results),
                conclusion=self.utils.draw_conclusion(tech_results),
                confidence=self.utils.calculate_confidence(tech_results),
                sources=tech_results
            ))
            
            # Layer 2: Practical Implementation
            blog_results = await self._search_blog_db(topic)
            layers.append(ReasoningLayer(
                name="Practical Implementation",
                description="Real-world applications and considerations",
                thought_process=self.utils.synthesize_thoughts(blog_results),
                conclusion=self.utils.draw_conclusion(blog_results),
                confidence=self.utils.calculate_confidence(blog_results),
                sources=blog_results
            ))
            
            # Layer 3: Evolution and Context
            temporal_results = await self._search_temporal_db(topic)
            layers.append(ReasoningLayer(
                name="Temporal Context",
                description="How thinking on this topic has evolved",
                thought_process=self.utils.synthesize_thoughts(temporal_results),
                conclusion=self.utils.draw_conclusion(temporal_results),
                confidence=self.utils.calculate_confidence(temporal_results),
                sources=temporal_results
            ))
            
            return layers
        except Exception as e:
            print(f"Error in layered analysis: {str(e)}")
            return []

    async def query(self, user_input: str, stream_handler: StreamingCallbackHandler = None) -> Dict:
        """Process a user query through the agent with streaming support"""
        try:
            # Configure callbacks
            callbacks = [stream_handler] if stream_handler else []
            
            # Get agent's response with streaming
            agent_response = await self.agent.ainvoke(
                {"input": user_input},
                config={"callbacks": callbacks}
            )
            
            # Perform layered analysis
            analysis = await self._layered_analysis(user_input)
            
            return {
                "response": agent_response['output'],
                "context": {
                    "analysis": analysis,
                    "sources": {
                        "technical": await self._search_technical_db(user_input),
                        "blog": await self._search_blog_db(user_input),
                        "temporal": await self._search_temporal_db(user_input)
                    }
                }
            }
        except Exception as e:
            print(f"Error during query processing: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }
        
    async def get_response_type(self, message: str) -> str:
           """Determine if a message requires knowledge base access"""
           try:
               prompt = f"""Is this a conversational message or one requiring technical knowledge?
               Message: {message}
               Reply with either CONVERSATIONAL or NEEDS_KNOWLEDGE."""
               
               response = await self.llm.ainvoke(
                   [HumanMessage(content=prompt)]
               )
               return response.content.strip().upper()
           except Exception as e:
               print(f"Error determining response type: {str(e)}")
               return "NEEDS_KNOWLEDGE"


@cl.on_chat_start
async def on_chat_start():
    """Initialize the VitalikAgent when a new chat starts"""
    try:
        print("Initializing VitalikAgent...")
        agent = VitalikAgent()  # Initialize with default persona path
        cl.user_session.set("agent", agent)
        print("Agent initialized successfully!")
        
        
    except Exception as e:
        error_msg = f"Error initializing agent: {str(e)}"
        print(error_msg)
        await cl.Message(content=error_msg, author="vitalikAI").send()


import chainlit as cl

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
    """Handle incoming messages"""
    agent = cl.user_session.get("agent")
    if not agent:
        await cl.Message(content="Agent not initialized. Please restart the chat.", author="vitalikAI").send()
        return

    msg = cl.Message(content="", author="vitalikAI")
    
    try:
        # First, check if this is a conversational query that doesn't need knowledge base access
        conversational_prompt = f"""Determine if this is a conversational message that doesn't require technical knowledge:
        Message: {message.content}
        
        If this is a simple greeting, personal question, a general question like 'who made you'S or general conversation that doesn't require technical knowledge,
        return CONVERSATIONAL. Otherwise, return NEEDS_KNOWLEDGE.
        
        Just return one word: CONVERSATIONAL or NEEDS_KNOWLEDGE."""

        # Get classification of message type
        classification_response = await agent.llm.ainvoke(
            [HumanMessage(content=conversational_prompt)]
        )
        is_conversational = "CONVERSATIONAL" in classification_response.content.upper()

        if is_conversational:
            # Handle conversational queries with a simpler prompt
            conversation_prompt = f"""As {agent.persona['name']}, {agent.persona['role']}, respond naturally to this conversational message:

            Message: {message.content}

            Remember:
            1. Be warm and engaging while maintaining my characteristic analytical style
            2. Keep responses concise for simple queries
            3. Stay true to my personality but don't overanalyze simple exchanges"""

            stream_handler = StreamingCallbackHandler(msg)
            response = await agent.agent.ainvoke(
                {"input": conversation_prompt},
                config={"callbacks": [stream_handler]}
            )

        else:
            # For technical/knowledge-based queries, use the full context gathering
            # Gather context from knowledge bases
            technical_context = await agent._search_technical_db(message.content)
            blog_context = await agent._search_blog_db(message.content)
            temporal_context = await agent._search_temporal_db(message.content)
            
            # Craft prompt using persona and gathered context
            technical_prompt = f"""As {agent.persona['name']}, {agent.persona['role']}, analyze this question using my characteristic approach:

            Technical Knowledge:
            {format_context(technical_context)}

            Blog Perspectives:
            {format_context(blog_context)}

            Historical Context:
            {format_context(temporal_context)}

            Question: {message.content}

            Respond in my distinctive style:
            1. Start with first principles thinking
            2. Consider technical, economic, and social implications
            3. Reference relevant research and past writings when applicable
            4. Maintain philosophical and analytical depth
            5. Use clear, precise language with technical accuracy
            6. Include mathematical or formal logic when relevant
            7. Address potential counterarguments
            8. Consider long-term implications
            
            Do this thinking internally to respond. The final reponse should not be generated by taking this as a template. This is just the thinking pattern guide not an answer template.
            """

            stream_handler = StreamingCallbackHandler(msg)
            response = await agent.agent.ainvoke(
                {"input": technical_prompt},
                config={"callbacks": [stream_handler]}
            )

        await msg.send()

    except Exception as e:
        await cl.Message(content=f"Error processing message: {str(e)}", author="Agent Vitalik").send()

if __name__ == "__main__":
    asyncio.run(cl.start())