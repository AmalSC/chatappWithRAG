import requests
import os

AGENT_API_URL = "http://localhost:11435/api/chat"   # or use env var


def call_local_agent(prompt: str):
    """
    Calls your local agent model through the API.
    Input: plain prompt text
    Returns: parsed JSON response from agent
    """

    body = {
        "model": "gemma3:1b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_ctx": 2048
        }
    }

    try:
        print("kkpp 12")
        response = requests.post(AGENT_API_URL, json=body)
        print("kkpp 13",response)
        agent_output = response.json()
        bot_response = agent_output.get('agent_response', agent_output.get('message', {}).get('content', 'Fallback response'))
        return {"bot_response":bot_response}
    except Exception as e:
        return {"error": str(e)}
