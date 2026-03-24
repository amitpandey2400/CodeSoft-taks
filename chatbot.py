"""Simple rule-based chatbot.

Run:
    python chatbot.py
"""

from __future__ import annotations

import datetime as dt
import re


def get_response(user_input: str) -> str:
    """Return a chatbot response based on simple rules and pattern matching."""
    text = user_input.strip().lower()

    if not text:
        return "Please type something so I can help."

    if re.search(r"\b(hi|hello|hey|hola)\b", text):
        return "Hello! How can I help you today?"

    if re.search(r"\b(name|who are you)\b", text):
        return "I am a simple rule-based chatbot."

    if re.search(r"\b(help|what can you do|commands)\b", text):
        return (
            "I can respond to greetings, tell date/time, answer basic questions, "
            "and end the chat when you type 'bye'."
        )

    if re.search(r"\b(time|current time)\b", text):
        current_time = dt.datetime.now().strftime("%I:%M %p")
        return f"Current time is {current_time}."

    if re.search(r"\b(date|today)\b", text):
        current_date = dt.datetime.now().strftime("%d %B %Y")
        return f"Today's date is {current_date}."

    if re.search(r"\b(how are you)\b", text):
        return "I am doing great. Thanks for asking!"

    if re.search(r"\b(thank you|thanks)\b", text):
        return "You're welcome!"

    if re.search(r"\b(bye|exit|quit)\b", text):
        return "Goodbye! Have a nice day."

    return "Sorry, I do not understand that yet. Try asking something else."


def chat() -> None:
    """Start an interactive chatbot session."""
    print("Rule-Based Chatbot")
    print("Type 'bye' to end the chat.")

    while True:
        user_input = input("You: ")
        response = get_response(user_input)
        print(f"Bot: {response}")

        if re.search(r"\b(bye|exit|quit)\b", user_input.strip().lower()):
            break


if __name__ == "__main__":
    chat()