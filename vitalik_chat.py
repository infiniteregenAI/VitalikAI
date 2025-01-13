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
from models import ReasoningLayer
from utils import VitalikUtils

class VitalikAgent:
    def _format_nested(self, template: str, data_dict: Dict) -> str:
        """Format string with nested dictionary access"""
        try:
            def resolve_key(key_path):
                """Resolve nested dictionary keys"""
                parts = key_path.split('.')
                value = data_dict
                for part in parts:
                    if '[' in part:  # Handle array access
                        array_name, index = part.split('[')
                        index = int(index.rstrip(']'))
                        value = value[array_name][index]
                    else:
                        value = value[part]
                return str(value)

            # Replace all placeholders
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
                
                # Validate required fields
                required_fields = ['name', 'role', 'system_prompt_template']
                missing_fields = [field for field in required_fields if field not in self.persona]
                if missing_fields:
                    raise ValueError(f"Missing required fields in persona.json: {missing_fields}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Persona file not found at {persona_path}. Please ensure persona.json exists in the current directory.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {persona_path}. Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading persona file: {str(e)}")
        
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                stream=True
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
            
            # Get collections
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
        """Search technical knowledge base"""
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
        """Search blog knowledge base"""
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
        """Search temporal knowledge base"""
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
        """Create tools for the agent to use"""
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
        """Create the agent with custom prompt"""
        try:
            # Create a simplified system prompt for initial testing
            simple_prompt = f"""I am {self.persona['name']}, {self.persona['role']}. 
            I approach problems from first principles, combining technical depth with philosophical considerations."""
            
            # Use simplified prompt for now
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
        """Perform layered analysis of a topic"""
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

    async def query(self, user_input: str) -> Dict:
        """Process a user query through the agent"""
        try:
            # Get agent's response with existing chat history
            agent_response = await self.agent.ainvoke({
                "input": user_input
            })
            
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

if __name__ == "__main__":
    async def main():
        try:
            print("Initializing VitalikAgent...")
            agent = VitalikAgent()
            print("Agent initialized successfully!")
            
            print("\nProcessing query...")
            result = await agent.query(
                "What are your thoughts on Ethereum scaling solutions?"
            )
            
            print("\nResponse:", result["response"])
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            print("\nFull traceback:")
            print(traceback.format_exc())
    
    asyncio.run(main())