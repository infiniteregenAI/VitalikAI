# VitalikAI

An attempt to replicate Vitalik Buterin's thought process, knowledge, and communication style in the form of an AI agent. The system leverages Vitalik's technical papers, blog posts, and historical content to generate responses and insights that align with his characteristic approach to blockchain, cryptography, and technological philosophy.

## System Architecture

### Base VitalikAI System
The core system emulates Vitalik's multi-layered thinking process by combining technical depth with philosophical considerations:

```mermaid
graph TD
    A[User Query] --> B[VitalikAI Core]
    B --> C{Knowledge Retrieval}
    
    C -->|Technical Knowledge| D[Technical Papers DB]
    C -->|Blog Insights| E[Blog Posts DB]
    C -->|Historical Context| F[Temporal Content DB]
    
    D --> G[Technical Analysis Layer]
    E --> H[Practical Implementation Layer]
    F --> I[Historical Evolution Layer]
    
    G --> J[Thought Synthesizer]
    H --> J
    I --> J
    
    J --> K[Response Generation]
    
    subgraph "Vitalik's Persona"
        L[First Principles Thinking]
        M[Technical-Philosophical Balance]
        N[Communication Style]
    end
    
    L --> K
    M --> K
    N --> K
```

### VitalikAI Tweet Generator
An extension that generates Vitalik-style insights in tweet format:

```mermaid
graph TD
    A[Tweet Generation] --> B[Topic Extraction]
    B -->|Sample Knowledge Bases| C[Vector DBs]
    C --> D[Topic Analysis]
    
    D --> E{Style Selection}
    E -->|Choose Approach| F[Style Framework]
    
    F -->|Format| G[Tweet Generation]
    
    subgraph "Writing Styles"
        H[Technical Deep Dive]
        I[Philosophical Reflection]
        J[Future Projection]
        K[Current Analysis]
        L[Historical Perspective]
    end
    
    F --> H
    F --> I
    F --> J
    F --> K
    F --> L
    
    G --> M[Knowledge Integration]
    M --> N[Final Tweet]
    
    subgraph "Authenticity Checks"
        O[Technical Accuracy]
        P[Style Matching]
        Q[Context Alignment]
    end
    
    M --> O
    M --> P
    M --> Q
```

# Running FastAPI:
```bash
python -m venv venv
```
```bash
.\venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
```bash
uvicorn api.conversation:app --reload
```