import requests

url = "http://localhost:1234/v1/chat/completions"

payload = {
    "model": "Mistral-7B-Instruct-v0.3-Q4_K_M",
    "messages": [{"role": "user", "content": "What is the capital of Quetta?"}]
}

response = requests.post(url, json=payload)
data = response.json()

reply = data['choices'][0]['message']['content']

print("Mistral response:",reply)
