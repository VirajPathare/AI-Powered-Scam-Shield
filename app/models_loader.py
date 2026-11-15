import os
from pathlib import Path
from typing import Optional, Tuple, Any
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    DistilBertTokenizerFast, DistilBertForSequenceClassification
)
import dotenv

# Load .env
REPO_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(REPO_ROOT / ".env")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_path(env_var: str, default: str) -> Path:
    raw = os.getenv(env_var, default)
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p

TEXT_MODEL_DIR = resolve_path("MM_TEXT_MODEL_DIR", "backend/models/text/bert_finetuned")
URL_MODEL_DIR = resolve_path("MM_URL_MODEL_DIR", "backend/models/url/distilbert_url")
IMAGE_MODEL_PATH = resolve_path("MM_IMAGE_MODEL_PATH", "backend/models/image/best_model.pth")

IMG_SIZE = int(os.getenv("MM_IMG_SIZE", "224"))
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class ModelRegistry:
    def __init__(self):
        self.text_tokenizer = None
        self.text_model = None
        self.url_tokenizer = None
        self.url_model = None
        self.image_model = None
        self.initialized = False

    def load_all(self):
        if self.initialized:
            return
        self.load_text()
        self.load_url()
        self.load_image()
        self.initialized = True

    def load_text(self):
        path_str = str(TEXT_MODEL_DIR)
        if os.path.isdir(path_str):
            try:
                print(f"[models_loader] Loading text model from {path_str}...")
                self.text_tokenizer = BertTokenizerFast.from_pretrained(path_str)
                self.text_model = BertForSequenceClassification.from_pretrained(path_str)
                self.text_model.to(DEVICE)
                self.text_model.eval()
                print("[models_loader] Text model loaded successfully.")
                return
            except Exception as e:
                print(f"[models_loader] Failed to load custom text model: {e}")
        print(f"[models_loader] Custom text model not found at {path_str}. Using heuristic analyzer fallback.")

    def load_url(self):
        path_str = str(URL_MODEL_DIR)
        if os.path.isdir(path_str):
            try:
                print(f"[models_loader] Loading URL model from {path_str}...")
                self.url_tokenizer = DistilBertTokenizerFast.from_pretrained(path_str)
                self.url_model = DistilBertForSequenceClassification.from_pretrained(path_str)
                self.url_model.to(DEVICE)
                self.url_model.eval()
                print("[models_loader] URL model loaded successfully.")
                return
            except Exception as e:
                print(f"[models_loader] Failed to load custom URL model: {e}")
        print(f"[models_loader] Custom URL model not found at {path_str}. Using heuristic analyzer fallback.")

    def load_image(self):
        path_str = str(IMAGE_MODEL_PATH)
        if os.path.isfile(path_str):
            try:
                print(f"[models_loader] Loading image model from {path_str}...")
                model = models.resnet50(weights=None)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, 2))
                sd = torch.load(path_str, map_location=DEVICE)
                if isinstance(sd, dict):
                    if "state_dict" in sd:
                        sd = sd["state_dict"]
                    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
                    model.load_state_dict(new_sd)
                else:
                    model.load_state_dict(sd)
                model.to(DEVICE)
                model.eval()
                self.image_model = model
                print("[models_loader] Image model loaded successfully.")
                return
            except Exception as e:
                print(f"[models_loader] Failed to load custom image model: {e}")
        print(f"[models_loader] Custom image model not found at {path_str}. Using heuristic analyzer fallback.")

registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    if not registry.initialized:
        registry.load_all()
    return registry
