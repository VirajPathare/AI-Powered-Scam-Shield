import os
import re
from io import BytesIO
from typing import Optional, Dict, Any
from PIL import Image
import torch
import requests
from app.models_loader import get_registry, img_transform, DEVICE

USE_OCR = os.getenv("MM_USE_OCR", "1") == "1"
N8N_OCR_ENDPOINT = os.getenv("MM_N8N_OCR_ENDPOINT", "http://localhost:5678/webhook/ocr")
TESSERACT_CMD = os.getenv("MM_TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
N8N_TIMEOUT = int(os.getenv("MM_N8N_TIMEOUT", "5"))
DEFAULT_ALPHA = float(os.getenv("MM_FUSION_ALPHA", "0.80"))
URL_BINARY_WEIGHT = float(os.getenv("MM_URL_BINARY_WEIGHT", "0.5"))

# ---------------- OCR Helpers ----------------
def ocr_n8n(image_bytes: bytes) -> Optional[str]:
    try:
        files = {"file": ("img.png", image_bytes, "image/png")}
        r = requests.post(N8N_OCR_ENDPOINT, files=files, timeout=N8N_TIMEOUT)
        if r.status_code == 200:
            j = r.json()
            return j.get("ocr_text") or j.get("text") or j.get("ocr") or None
    except Exception:
        pass
    return None

def ocr_local(image_bytes: bytes) -> Optional[str]:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        txt = pytesseract.image_to_string(img)
        return txt if txt and txt.strip() else None
    except Exception:
        return None

def get_ocr_text_from_bytes(image_bytes: bytes) -> Optional[str]:
    if not USE_OCR:
        return None
    txt = ocr_n8n(image_bytes)
    if txt and txt.strip():
        return txt.strip()
    txt = ocr_local(image_bytes)
    if txt and txt.strip():
        return txt.strip()
    return None

# ---------------- Heuristic Fallback Scanners ----------------
def heuristic_text_score(text: str) -> float:
    t = text.lower()
    score = 0.05
    suspicious_patterns = [
        (r"\b(urgent|immediate action|suspended|verify your account|unauthorized|blocked)\b", 0.35),
        (r"\b(click here|log in now|reset password|claim reward|lottery|prize|winner)\b", 0.30),
        (r"\b(wire transfer|gift card|crypto|bitcoin|western union|ssn|social security)\b", 0.35),
        (r"\b(bank of|paypal|netflix|apple id|microsoft|amazon|google account)\b", 0.20),
        (r"https?://[^\s]+", 0.15),
        (r"\b(dear customer|dear user|dear client|valued customer)\b", 0.15),
        (r"\b(24 hours|limited time|immediately|expire|penalty)\b", 0.20)
    ]
    for pat, weight in suspicious_patterns:
        if re.search(pat, t):
            score += weight
    return float(min(0.98, max(0.02, score)))

def heuristic_url_score(url: str) -> Dict[str, float]:
    u = url.lower()
    phish_score = 0.05
    malware_score = 0.02

    # Suspicious keywords in URL
    if any(k in u for k in ["login", "verify", "secure", "update", "banking", "wallet", "signin", "auth", "confirm", "free", "gift"]):
        phish_score += 0.35
    # Brand impersonation checks
    if any(b in u for b in ["paypal", "netflix", "apple", "microsoft", "google", "amazon", "chase", "wellsfargo"]) and not any(u.endswith(f".{b}.com") or f".{b}.com/" in u for b in ["paypal", "netflix", "apple", "microsoft", "google", "amazon", "chase", "wellsfargo"]):
        phish_score += 0.40
    # Suspicious TLDs
    if re.search(r"\.(xyz|top|work|click|cf|tk|ml|gq|ga|buzz|rest|live|fit|bar)\b", u):
        phish_score += 0.25
        malware_score += 0.15
    # Raw IP address used as host
    if re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", u):
        phish_score += 0.45
        malware_score += 0.20
    # Malware extensions
    if re.search(r"\.(exe|scr|apk|vbs|bat|cmd|ps1|iso|dmg)$", u.split("?")[0]):
        malware_score += 0.70
        phish_score += 0.20
    # Abnormal length or excess hyphens/subdomains
    if u.count(".") > 3 or u.count("-") > 3:
        phish_score += 0.20

    phish_score = min(0.95, max(0.02, phish_score))
    malware_score = min(0.95, max(0.01, malware_score))
    benign_score = max(0.01, 1.0 - (phish_score + malware_score))
    total = phish_score + malware_score + benign_score

    return {
        "benign": round(float(benign_score / total), 4),
        "phishing": round(float(phish_score / total), 4),
        "malware": round(float(malware_score / total), 4),
    }

from app.dataset_intel import get_dataset_intel

# ---------------- Model Predictors ----------------
def predict_text_probs(text: Optional[str]) -> Dict[str, float]:
    if not text or not str(text).strip():
        return {"benign": 0.99, "phishing": 0.01}
    
    # Real-time check against uploaded Phishing_Email dataset signatures
    intel = get_dataset_intel().check_text(text)
    intel_boost = 0.25 if (intel and intel.get("matched")) else 0.0

    reg = get_registry()
    if reg.text_tokenizer and reg.text_model:
        try:
            enc = reg.text_tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                logits = reg.text_model(**enc).logits
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            if len(probs) >= 2:
                p_phish = min(0.99, float(probs[1]) + intel_boost)
                return {"benign": round(1.0 - p_phish, 4), "phishing": round(p_phish, 4)}
        except Exception as e:
            print(f"[predictors] Text model inference error: {e}")
    
    # Fallback heuristic + dataset intelligence
    p_phish = heuristic_text_score(text)
    if intel_boost > 0:
        p_phish = min(0.98, p_phish + intel_boost)
    return {"benign": round(1.0 - p_phish, 4), "phishing": round(p_phish, 4)}

def predict_url_probs(url: Optional[str]) -> Dict[str, float]:
    if not url or not str(url).strip():
        return {"benign": 0.99, "phishing": 0.01, "malware": 0.0}

    # Real-time lookup against uploaded URL threat datasets (malicious_phish.csv, top-1m.csv)
    intel = get_dataset_intel().check_url(url)
    if intel:
        if intel.get("matched"):
            ttype = intel.get("threat_type", "phishing")
            if ttype == "malware":
                return {"benign": 0.01, "phishing": 0.15, "malware": 0.84}
            return {"benign": 0.02, "phishing": 0.95, "malware": 0.03}
        elif intel.get("threat_type") == "benign":
            # Verified top benign domain
            h_score = heuristic_url_score(url)
            if h_score.get("phishing", 0.0) < 0.40:
                return {"benign": 0.96, "phishing": 0.03, "malware": 0.01}

    reg = get_registry()
    if reg.url_tokenizer and reg.url_model:
        try:
            enc = reg.url_tokenizer(url, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                logits = reg.url_model(**enc).logits
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            if len(probs) == 3:
                return {"benign": round(float(probs[0]), 4), "phishing": round(float(probs[1]), 4), "malware": round(float(probs[2]), 4)}
            elif len(probs) == 2:
                return {"benign": round(float(probs[0]), 4), "phishing": round(float(probs[1]), 4), "malware": 0.0}
        except Exception as e:
            print(f"[predictors] URL model inference error: {e}")
    
    # Fallback heuristic
    return heuristic_url_score(url)

def predict_image_probs(image_bytes: bytes) -> Dict[str, float]:
    reg = get_registry()
    if reg.image_model:
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            x = img_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = reg.image_model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            if len(probs) >= 2:
                return {"benign": round(float(probs[0]), 4), "phishing": round(float(probs[1]), 4)}
        except Exception as e:
            print(f"[predictors] Image model inference error: {e}")
    # Fallback: check OCR text if available
    ocr_txt = get_ocr_text_from_bytes(image_bytes)
    if ocr_txt:
        p_phish = heuristic_text_score(ocr_txt)
        return {"benign": round(1.0 - p_phish, 4), "phishing": round(p_phish, 4)}
    return {"benign": 0.85, "phishing": 0.15}

# ---------------- Multimodal Fusion ----------------
def url_multiclass_to_binary(url_probs: Dict[str, float]) -> float:
    p_phish = url_probs.get("phishing", 0.0)
    p_malware = url_probs.get("malware", 0.0)
    return float(min(1.0, p_phish + p_malware))

def fuse_all(
    text_p: Optional[float],
    url_p_bin: Optional[float],
    img_p: Optional[float],
    ocr_p: Optional[float] = None,
    alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    modalities = {"text": text_p, "url": url_p_bin, "img": img_p, "ocr": ocr_p}
    text_signal = None
    if text_p is not None and ocr_p is not None:
        text_signal = max(text_p, ocr_p)
    elif text_p is not None:
        text_signal = text_p
    elif ocr_p is not None:
        text_signal = ocr_p

    available = {k: v for k, v in [("text", text_signal), ("url", url_p_bin), ("img", img_p)] if v is not None}
    if not available:
        return {"fused_prob": 0.0, "label": "benign", **modalities}

    if "text" in available:
        rem = 1.0 - alpha
        others = [k for k in available.keys() if k != "text"]
        if others:
            share = rem / len(others)
            fused = alpha * available["text"]
            for o in others:
                fused += share * available[o]
        else:
            fused = available["text"]
    else:
        fused = sum(available.values()) / len(available)

    fused = float(min(1.0, max(0.0, fused)))
    label = "phishing" if fused >= 0.5 else "benign"
    return {
        "fused_prob": round(fused, 4),
        "label": label,
        "probs": {
            "benign": round(1.0 - fused, 4),
            "phishing": round(fused, 4)
        },
        **modalities
    }
