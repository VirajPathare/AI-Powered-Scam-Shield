<div align="center">

  <img src="./frontend/public/frontend.png" alt="Scam Shield Application Interface" width="640" style="max-width:100%;border-radius:8px;"/>

  <br>

  <h1>AI-Powered Scam Shield: An Intelligent Cyber Fraud Detection System</h1>
  
  <p>
    <img alt="Build Status" src="https://img.shields.io/badge/build-passing-brightgreen">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.10%2B-blue">
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi&logoColor=white">
    <img alt="React" src="https://img.shields.io/badge/React-FrontEnd-61DAFB?logo=react&logoColor=black">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ML-orange?logo=pytorch">
    <img alt="Transformers" src="https://img.shields.io/badge/Transformers-HF-yellow?logo=huggingface&logoColor=black">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  </p>
  
  <p><em>A multimodal cyber fraud and scam detection platform utilizing screenshots, text, and URLs. Powered by BERT, DistilBERT, ResNet50, and OCR.</em></p>

</div>

---

## Project Details

| Attribute | Details |
|---|---|
| **Project Title** | AI-Powered Scam Shield: An Intelligent Cyber Fraud Detection System |
| **Type** | Software |
| **Technology** | Artificial Intelligence (AI), Data Analytics, Web Development |
| **Domain** | Cybersecurity |

## Project Description

The increasing use of digital platforms, online transactions, and social media has led to a significant rise in cyber fraud and online scams. Many users receive suspicious messages, misleading links, and fraudulent requests through various communication channels. However, identifying such threats can be challenging, especially for individuals unfamiliar with cybersecurity practices.

Existing solutions often focus on specific threat types and may not provide a unified approach to fraud detection, leaving users vulnerable to digital scams and potential financial losses. 

This proposed system aims to develop an AI-powered platform that assists users in recognizing potentially fraudulent content and provides timely insights to improve digital security awareness. By combining artificial intelligence and analytical techniques, the system helps users make safer decisions when interacting with online platforms.

---

## System Overview

A comprehensive multimodal cyber fraud and scam detection system that combines:

- **Text-based scam detection** using a fine-tuned BERT transformer model.
- **URL threat classification** with fine-tuned DistilBERT multiclass transformer classification.
- **Image-based fraud detection** leveraging a ResNet50 deep CNN with optional OCR.
- **Multimodal Fusion logic** to combine multiple fraud signals into a unified threat score.
- **FastAPI backend** integrated with a **React frontend**.
- **Chat mode** powered by an intelligent LLM for explainable fraud alerts.

The system provides a complete pipeline from training to evaluation to real-time inference with a production-ready UI.

---

## Model Performance

Evaluation results from comprehensive testing:

| Modality | Model | Accuracy | Precision | Recall | F1 Score |
|----------|-------|----------|-----------|--------|----------|
| **Image** | ResNet50 (25 epochs, best val: 83.30%) | 78.99% | 72.69% | 82.63% | 77.34% |
| **Text** | Fine-tuned BERT | 97.22% | 97.51% | 95.84% | 96.67% |
| **URL** | DistilBERT (3-class) | **99.50%** | 99.49% | 99.49% | 99.49% |

**Key Metrics:**
- Image model best validation accuracy: **83.30%**
- Text model ROC-AUC: **0.9961**
- URL model overall accuracy: **99.50%**

---

## Detailed URL Classification Results

**Test Dataset:** 22,182 samples  
**Overall Accuracy:** 99.50%  
**Macro F1:** 0.9949  
**Weighted F1:** 0.9950

### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 99.64% | 100.00% | 99.82% | 7,500 |
| Phishing | 99.18% | 99.43% | 99.30% | 7,182 |
| Malware | 99.65% | 99.05% | 99.35% | 7,500 |

**Generated Visualizations:**
- `runs/url_eval/roc_curves.png`
- `runs/url_eval/confusion_matrix.png`
- `runs/url_eval/confidence_histogram.png`

---

## Tech Stack

### Backend & Machine Learning
- **FastAPI** - High-performance async API framework
- **PyTorch** - Deep learning framework
- **Torchvision** - Computer vision models (ResNet50)
- **Transformers (HuggingFace)** - BERT and DistilBERT implementations
- **Scikit-learn** - ML utilities and metrics
- **Pandas & NumPy** - Data processing
- **Pytesseract** - OCR capabilities (optional)
- **Uvicorn** - ASGI server

### Frontend
- **React (Vite)** - Modern frontend framework
- **Tailwind CSS** - Utility-first styling
- **Shadcn UI** - Component library
- **Framer Motion** - Animation library

---

## Project Structure

### Backend Components

```text
backend/
├── app/
│   ├── main.py              # FastAPI routes and endpoints
│   ├── models_loader.py     # Model initialization and loading
│   ├── predictors.py        # Text/URL/Image inference logic
│   ├── intent.py            # Multimodal intent detection
│   └── llm_helpers.py       # LLM-based explanations
