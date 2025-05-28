import requests
import json

def send_request_to_ollama(prompt: str, base_url: str, model_name: str = "gemma3:12b", options: dict = None) -> dict:
    """
    Sends a request to the Ollama API and returns the response.

    Args:
        prompt: The prompt to send to the model.
        base_url: The base URL of the Ollama API.
        model_name: The name of the model to use.
        options: A dictionary of options for the Ollama API.

    Returns:
        A dictionary containing the response from the Ollama API.
    """
    api_url = f"{base_url}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON response"}

if __name__ == '__main__':
    # Example usage (optional, for testing)
    # Replace with your actual Ollama API base URL if different
    ollama_base_url = "http://localhost:11434"
    
    # Example prompt
    example_prompt = "What is the capital of France?"
    
    # Example options (optional)
    example_options = {"temperature": 0.7}
    
    print(f"Sending request to Ollama with prompt: '{example_prompt}'")
    response_data = send_request_to_ollama(example_prompt, ollama_base_url, options=example_options)
    
    if "error" in response_data:
        print(f"Error: {response_data['error']}")
    elif "response" in response_data:
        print(f"Ollama's response: {response_data['response']}")
    else:
        print(f"Full response data: {response_data}")
