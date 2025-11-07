
import requests
import json

# The endpoint for the second LLM, as provided in the curl command.
LLM_API_URL = "http://140.112.31.177:1357/v1/chat/completions"
LLM_MODEL_NAME = "Intelligent-Internet/II-Medical-8B"

def query_medical_llm(prompt: str) -> str:
    """
    Sends a prompt to the second, medical LLM and returns its response.

    Args:
        prompt: The input text to send to the model.

    Returns:
        The content of the model's response as a string.
        Returns an error message if the request fails.
    """
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_json = response.json()
        
        # Extract the content from the first choice in the response.
        if response_json.get("choices") and len(response_json["choices"]) > 0:
            message = response_json["choices"][0].get("message", {})
            content = message.get("content", "Error: Could not find content in response.")
            return content
        else:
            return "Error: 'choices' field is missing or empty in the response."

    except requests.exceptions.RequestException as e:
        return f"Error making request to LLM: {e}"
    except json.JSONDecodeError:
        return f"Error: Could not decode JSON response from LLM. Response text: {response.text}"

if __name__ == '__main__':
    # Example usage for testing this module directly
    test_prompt = "What are the symptoms of diabetes?"
    print(f"Testing with prompt: '{test_prompt}'")
    response_content = query_medical_llm(test_prompt)
    print("--- LLM Response ---")
    print(response_content)
    print("--------------------")
