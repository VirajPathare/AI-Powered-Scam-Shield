import os
import json
import requests
from typing import Dict, Any, Optional

LLM_PROVIDER = os.getenv("MM_LLM_PROVIDER", "none")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
MM_HF_MODEL = os.getenv("MM_HF_MODEL", "google/gemma-2-2b-it")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", "")

try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_AVAILABLE = True
except Exception:
    HF_CLIENT_AVAILABLE = False

def call_hf(prompt: str) -> Optional[str]:
    if not HUGGINGFACE_API_KEY or not HF_CLIENT_AVAILABLE:
        return None
    try:
        client = InferenceClient(model=MM_HF_MODEL, token=HUGGINGFACE_API_KEY)
        resp = client.chat.completions.create(
            model=MM_HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.2
        )
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[llm_helpers] HF call error: {e}")
    return None

def call_gemini(prompt: str) -> Optional[str]:
    if not GEMINI_API_KEY or not GEMINI_ENDPOINT:
        return None
    try:
        headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
        r = requests.post(GEMINI_ENDPOINT, headers=headers, json={"prompt": prompt}, timeout=20)
        if r.status_code == 200:
            j = r.json()
            for k in ("output", "text", "response", "content", "candidates"):
                if k in j:
                    val = j[k]
                    return val[0] if isinstance(val, list) else str(val)
    except Exception as e:
        print(f"[llm_helpers] Gemini call error: {e}")
    return None

def make_llm_explanation(summary: Dict[str, Any], ocr_text: Optional[str] = None) -> str:
    prompt = f"""You are CyberGuardian, a cybersecurity assistant.
Explain this scan result concisely:
{json.dumps(summary, indent=2)}
OCR text: {ocr_text or 'None'}
Provide a 1-sentence verdict, 3 quick bullet points explaining why, and 1 safety tip."""

    if LLM_PROVIDER == "hf":
        res = call_hf(prompt)
        if res:
            return res
    elif LLM_PROVIDER == "gemini":
        res = call_gemini(prompt)
        if res:
            return res

    # Structured heuristic explanation
    label = summary.get("label", "benign").upper()
    probs = summary.get("probs", {})
    p_phish = probs.get("phishing", 0.0)
    p_mal = probs.get("malware", 0.0)
    threat_prob = p_phish + p_mal

    lines = []
    if label == "PHISHING":
        lines.append(f"**High Risk Phishing Detected ({threat_prob * 100:.1f}% confidence).**")
        lines.append("\n**Key Findings:**")
        lines.append("• Suspicious indicators found matching known phishing and credential theft patterns.")
        lines.append("• Deceptive elements or artificial urgency detected in the provided content.")
        if ocr_text:
            lines.append(f"• Screenshot text analyzed: \"{ocr_text[:80]}...\"")
        lines.append("\n**Recommended Action:**")
        lines.append("Do NOT click any links, open attachments, or input personal credentials. Report and delete this immediately.")
    elif label == "MALWARE":
        lines.append(f"**Malicious Software Warning ({p_mal * 100:.1f}% confidence).**")
        lines.append("\n**Key Findings:**")
        lines.append("• Target destination appears associated with malware distribution or drive-by downloads.")
        lines.append("• High-risk domain characteristics and executable references detected.")
        lines.append("\n**Recommended Action:**")
        lines.append("Block this destination immediately and do not attempt to download files.")
    else:
        lines.append(f"**Content Appears Benign ({probs.get('benign', 0.9) * 100:.1f}% confidence).**")
        lines.append("\n**Key Findings:**")
        lines.append("• No significant phishing triggers, deceptive domains, or manipulative urgency found.")
        lines.append("• Standard formatting consistent with regular legitimate communications.")
        lines.append("\n**Safety Tip:**")
        lines.append("Always verify sender addresses independently before sharing sensitive confidential information.")

    return "\n".join(lines)

def friendly_chat_reply(user_query: str) -> str:
    prompt = f"""You are CyberGuardian, a friendly AI cybersecurity advisor.
User query: "{user_query}"
Respond concisely and helpfully (2-3 sentences) with advice on staying safe or explaining how to scan messages/URLs."""

    if LLM_PROVIDER == "hf":
        res = call_hf(prompt)
        if res:
            return res
    elif LLM_PROVIDER == "gemini":
        res = call_gemini(prompt)
        if res:
            return res

    q = user_query.lower()
    if any(k in q for k in ["hello", "hi", "hey"]):
        return "Hello! I'm CyberGuardian, your AI security shield. Paste any suspicious text, URL, or upload a screenshot, and I'll analyze it for phishing and malware threats."
    if "who are you" in q:
        return "I am CyberGuardian, an AI-powered phishing and scam detection assistant that uses multimodal deep learning to protect you from online fraud."
    return "I can help verify suspicious messages, emails, URLs, and screenshots for phishing or fraud. Paste your query or upload an image anytime to run a security check!"
