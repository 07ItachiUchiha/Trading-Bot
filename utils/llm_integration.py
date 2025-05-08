import os
import requests

def get_llm_response(prompt, model=None):
    """
    Get response from OpenRouter LLM
    """
    model = model or os.getenv('LLM_MODEL', 'meta-llama/llama-3-8b-instruct')
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"LLM API error: {response.text}")
