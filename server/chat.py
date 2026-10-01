import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

if LLM_PROVIDER == "anthropic":
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    client = None

def get_intent_and_query(messages):
    last_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_msg = msg["content"]
            break
            
    if LLM_PROVIDER != "anthropic" or not client:
        return {"intent": "recommend", "name": None, "ingredients": [], "query_text": last_msg}
        
    prompt = f"""
You must understand the user's intent from the chat history. The user may be asking for recipe recommendations or something else.
Please output ONLY a JSON object with this exact schema:
{{
  "intent": "recommend" | "other",
  "name": "recipe name if provided, else null",
  "ingredients": ["ingredient1", "ingredient2"], // list of ingredients if provided
  "query_text": "the user's last message"
}}
Do not include any other text or markdown formatting. Output raw JSON only.
Chat history:
{json.dumps(messages, indent=2)}
"""
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        
        parsed = json.loads(content)
        intent = parsed.get("intent", "recommend")
        if intent not in ["recommend", "other"]:
            intent = "recommend"
        return {
            "intent": intent,
            "name": parsed.get("name"),
            "ingredients": parsed.get("ingredients", []),
            "query_text": parsed.get("query_text", last_msg)
        }
    except Exception as e:
        return {"intent": "recommend", "name": None, "ingredients": [], "query_text": last_msg}

def generate_reply(intent, messages, results):
    if intent == "other":
        return "I can only help you find recipes! Try telling me some ingredients you have or a dish you'd like to make."
        
    if LLM_PROVIDER != "anthropic" or not client:
        reply = "Here are some recommendations:\n"
        for r in results:
            reply += f"- {r['name']} (Score: {r['score']})\n"
        return reply
        
    results_context = json.dumps([{
        "name": r["name"],
        "ingredients": r["ingredients"]
    } for r in results], indent=2)
    
    system_prompt = """You are a helpful recipe recommendation assistant. 
Based on the retrieved recipes provided below, write 3-5 sentences recommending them to the user.
Rules:
- NEVER mention any recipe that is not in the provided list.
- Explain similarity using shared ingredients or dish type.
- Be honest that similarity is based on ingredients and dish type, not exact flavor.
- Do not invent facts, nutrition claims, or cooking instructions.

Retrieved recipes:
""" + results_context
    
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        reply = "Here are some recommendations:\n"
        for r in results:
            reply += f"- {r['name']} (Score: {r['score']})\n"
        return reply
