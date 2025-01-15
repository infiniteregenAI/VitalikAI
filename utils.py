from typing import List, Dict
from langchain_openai import ChatOpenAI

class VitalikUtils:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
    def format_results(self, results: Dict, db_type: str) -> List[Dict]:
        """Format search results with metadata"""
        return [{
            "content": doc,
            "metadata": meta,
            "type": db_type,
            "relevance_score": score
        } for doc, meta, score in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]
        
    def synthesize_thoughts(self, results: List[Dict]) -> str:
        """Synthesize thoughts based on search results"""
        prompt = f"""Given this information, what would I (Vitalik) think about this?
        Consider my writing style and typical approach to such topics.
        
        Information:
        {results}"""
        
        response = self.llm.predict(prompt)
        return response
        
    def draw_conclusion(self, results: List[Dict]) -> str:
        """Draw conclusions from search results"""
        prompt = f"""Based on these findings, what concrete insights would I (Vitalik) focus on?
        Frame it in my characteristic style of combining technical and philosophical perspectives.
        
        Findings:
        {results}"""
        
        response = self.llm.predict(prompt)
        return response
        
    def calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score based on result relevance"""
        if not results:
            return 0.0
        avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
        return 1 - (avg_relevance / 2)

    def get_classification_prompt(self, message: str) -> str:
        """Generate prompt for message classification"""
        return f"""Determine if this is a conversational message that doesn't require technical knowledge:
        Message: {message}
        
        If this is a simple greeting, personal question, or general conversation that doesn't require technical knowledge,
        return CONVERSATIONAL. Otherwise, return NEEDS_KNOWLEDGE.
        
        Just return one word: CONVERSATIONAL or NEEDS_KNOWLEDGE."""

    def get_conversational_prompt(self, message: str, persona_name: str, persona_role: str) -> str:
        """Generate prompt for conversational responses"""
        return f"""As {persona_name}, {persona_role}, respond naturally to this conversational message:

        Message: {message}

        Remember:
        1. Be warm and engaging while maintaining my characteristic analytical style
        2. Keep responses concise for simple queries
        3. Stay true to my personality but don't overanalyze simple exchanges"""

    def get_technical_prompt(self, message: str, technical_context: List[Dict], 
                           blog_context: List[Dict], temporal_context: List[Dict], 
                           persona: Dict) -> str:
        """Generate prompt for technical responses"""
        return f"""As {persona['name']}, {persona['role']}, analyze this question using my characteristic approach:

        Technical Knowledge:
        {self.format_context(technical_context)}

        Blog Perspectives:
        {self.format_context(blog_context)}

        Historical Context:
        {self.format_context(temporal_context)}

        Question: {message}

        Respond in my distinctive style:
        1. Start with first principles thinking
        2. Consider technical, economic, and social implications
        3. Reference relevant research and past writings when applicable
        4. Maintain philosophical and analytical depth
        5. Use clear, precise language with technical accuracy
        6. Include mathematical or formal logic when relevant
        7. Address potential counterarguments
        8. Consider long-term implications"""

    def format_context(self, context_results: List[Dict]) -> str:
        """Format context results for prompt input"""
        if not context_results:
            return "No specific historical context found for this topic."
            
        formatted = []
        for result in context_results:
            if isinstance(result, dict):
                content = result.get('content', '')
                if content:
                    formatted.append(f"- {content}")
                
        return "\n".join(formatted)