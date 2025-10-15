from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os, requests

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

#---------- 1. Orchestrator ----------
def classify_intent_llm(user_input: str) -> str:
    prompt = f"""
    Classify the following D&D player input into one of two categories:
    - narration
    - combat

    Input: "{user_input}"
    Output:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()

class UserInput(BaseModel):
    text: str

# ---------- 2. Combat Agent ----------
@app.post("/agent/combat")
def combat_agent(data: UserInput):
    user_input = data.text
    # GraphDB
    # RAG
    # 筛子
    # LLM
    result = f"⚔️ Combat agent received: {user_input}"
    return {"agent": "combat", "result": result}

# ---------- 3. Narrator Agent ----------
@app.post("/agent/narration")
def narrator_agent(data: UserInput):
    user_input = data.text
    # Finetune LLM
    # RAG
    # GraphDB
    result = f"📜 Narration agent received: {user_input}"
    return {"agent": "narration", "result": result}


@app.post("/orchestrate")
def orchestrate(data: UserInput):
    user_input = data.text
    intent = classify_intent_llm(user_input)

    if "combat" in intent:
        intent = "combat"
        response = combat_agent(data)
    else:
        intent = "narration"
        response = narrator_agent(data)

    print(f"🧭 Intent detected: {intent}")

    return {
        "orchestrator_intent": intent,
        "agent_response": response
    }