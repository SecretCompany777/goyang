import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Auto install dependencies jika belum ada
os.system("pip install transformers torch")

# Muat turun model AI
model_name = "TheBloke/Mistral-7B"  # Ganti dengan model yang sesuai
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_ai_response(text):
    inputs = tokenizer(text, return_tensors="pt")
    response = model.generate(**inputs)
    return tokenizer.decode(response[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("🤖 Chatbot AI (Tanpa Sensor) - Taip 'exit' untuk keluar")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot ditamatkan.")
            break
        bot_reply = get_ai_response(user_input)
        print("Bot:", bot_reply)