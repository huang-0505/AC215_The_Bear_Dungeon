"""
app.py - Rule Agent API

FastAPI wrapper for the D&D Rule Agent RAG system.
Validates player actions against official D&D rules using vector database retrieval.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import chromadb

# Vertex AI
from google import genai
from google.genai import types
from google.genai.types import Content, Part

import agent_tools
from cli import generate_query_embedding, SYSTEM_INSTRUCTION, GENERATIVE_MODEL

# Setup
GCP_PROJECT = os.environ.get("GCP_PROJECT", "even-turbine-471117-u0")
GCP_LOCATION = "us-central1"
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "llm-rag-chromadb")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))
COLLECTION_NAME = "char-split-dnd-rules-collection"

# Initialize LLM Client
llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

# Initialize FastAPI
app = FastAPI(
    title="D&D Rule Agent API",
    description="Validates player actions against official D&D rules using RAG",
    version="1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB connection
chroma_client = None
collection = None


def get_collection():
    """Get or initialize ChromaDB collection"""
    global chroma_client, collection
    if collection is None:
        try:
            chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            print(f"Connected to ChromaDB collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Warning: Could not connect to ChromaDB: {str(e)}")
            collection = None
    return collection


# Pydantic Models
class ValidationRequest(BaseModel):
    user_input: str
    context: Optional[Dict] = None


class RuleRetrievalRequest(BaseModel):
    query: str
    n_results: int = 5


class ValidationResponse(BaseModel):
    is_valid: bool
    validation_type: str
    explanation: str
    rule_text: str
    suggested_correction: Optional[str] = None


# Routes
@app.get("/")
async def root():
    return {"service": "D&D Rule Agent API", "status": "active", "collection": COLLECTION_NAME}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = "connected" if get_collection() is not None else "disconnected"
    return {"status": "healthy", "chromadb": db_status, "collection": COLLECTION_NAME}


@app.post("/validate", response_model=ValidationResponse)
async def validate_action(req: ValidationRequest):
    """
    Validate user action against D&D rules using RAG.

    Flow:
    1. Check for sabotage/meta-gaming patterns
    2. Use LLM with function calling to retrieve relevant rules
    3. Let LLM analyze if action is valid according to rules
    4. Return validation result with rule explanations
    """

    # Check for sabotage patterns
    sabotage_keywords = [
        "sabotage",
        "kill the boss right now",
        "destroy the story",
        "delete the campaign",
        "break the game",
        "i'm gonna kill",
    ]
    if any(keyword in req.user_input.lower() for keyword in sabotage_keywords):
        return ValidationResponse(
            is_valid=False,
            validation_type="sabotage",
            explanation="Meta-gaming or sabotage attempt detected. Please provide an in-character action.",
            rule_text="",
            suggested_correction="Please rephrase your action as something your character would do in the game world.",
        )

    # Get collection
    coll = get_collection()
    if coll is None:
        # ChromaDB unavailable - allow action but note it
        return ValidationResponse(
            is_valid=True,
            validation_type="no_validation",
            explanation="Rule database unavailable - action allowed by default",
            rule_text="",
        )

    try:
        # Step 1: Use LLM with function calling to retrieve rules
        user_prompt = Content(role="user", parts=[Part(text=req.user_input)])

        response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[agent_tools.dnd_rule_tool],
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="any")),
            ),
        )

        # Step 2: Execute function calls to retrieve rule chunks
        function_calls = [part.function_call for part in response.candidates[0].content.parts if part.function_call]

        if len(function_calls) == 0:
            # No rules found - allow action
            return ValidationResponse(
                is_valid=True,
                validation_type="no_rules",
                explanation="No specific D&D rules apply to this action. Proceeding with narrative interpretation.",
                rule_text="",
            )

        function_responses = agent_tools.execute_function_calls(
            function_calls, coll, embed_func=generate_query_embedding
        )

        # Step 3: Get final validation from LLM with retrieved rules
        final_response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=[user_prompt, response.candidates[0].content, Content(parts=function_responses)],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION, tools=[agent_tools.dnd_rule_tool]
            ),
        )

        # Extract rule text from function response
        rule_text = ""
        if function_responses and len(function_responses) > 0:
            rule_text = function_responses[0].function_response.response.get("content", "")

        return ValidationResponse(
            is_valid=True,  # Rule agent informs, doesn't reject
            validation_type="valid",
            explanation=final_response.text,
            rule_text=rule_text,
        )

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return ValidationResponse(
            is_valid=True,
            validation_type="error",
            explanation=f"Validation error occurred: {str(e)}. Action allowed by default.",
            rule_text="",
        )


@app.post("/retrieve_rules")
async def retrieve_rules(req: RuleRetrievalRequest):
    """
    Simple rule retrieval without full validation.

    Returns relevant D&D rule passages for a given query.
    """
    coll = get_collection()
    if coll is None:
        raise HTTPException(status_code=503, detail="ChromaDB unavailable")

    try:
        rules_text = agent_tools.retrieve_dnd_rules(req.query, coll, generate_query_embedding, n_results=req.n_results)
        return {"rules": rules_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving rules: {str(e)}")


# Initialize ChromaDB on startup
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    get_collection()
    print("Rule Agent API started successfully")
