import os
import re
from typing import Optional

def extract_single_url(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r'[<>\(\)\[\]\{\},;:]+', ' ', text)
    full = re.search(r'(https?://[^\s]+)', cleaned)
    if full:
        return full.group(1).rstrip(".,;:!?)(")
    dom = re.search(r'\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(/[^\s]*)?\b', cleaned)
    if dom:
        url = dom.group(0).rstrip(".,;:!?)(")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url
    return None

def detect_intent(user_query: str, has_file: bool = False) -> str:
    q = (user_query or "").strip().lower()

    if not q and has_file:
        return "image_check"

    extracted_url = extract_single_url(user_query)
    has_image_ref = any(w in q for w in ["screenshot", "image", "photo", "pic", "picture"])

    # If file is attached with text or URL -> combined mode
    if has_file and (extracted_url or len(q) > 15):
        return "combined"

    if has_file or (has_image_ref and not extracted_url and len(q) < 40):
        return "image_check"

    # URL check has priority if there's a URL and user is asking about it
    if extracted_url:
        if any(w in q for w in ["safe", "visit", "link", "url", "check", "legit", "is this", "open"]):
            return "url_check"
        if len(q.split()) <= 5:
            return "url_check"

    # Conversational / Chit-chat questions
    chat_patterns = [
        "hi", "hello", "hey", "howdy", "greetings",
        "who are you", "what are you", "what can you do", "help me",
        "how can you help", "how does this work", "how do you work",
        "what is phishing", "tell me about scams", "how to stay safe"
    ]
    if any(p in q for p in chat_patterns) and not any(k in q for k in ["urgent", "verify your account", "suspended", "password reset", "otp"]):
        return "chat"

    # Email / SMS indicators
    if any(k in q for k in ["subject:", "dear ", "account", "urgent", "verify", "bank", "otp", "password", "suspended", "wire transfer"]):
        return "text_message"

    # If long text block, treat as text message
    if len(q) > 40:
        return "text_message"

    # Short query fallback: if it sounds like a chat or question
    if "?" in q or any(w in q for w in ["why", "how", "what", "can you"]):
        return "chat"

    return "text_message"
