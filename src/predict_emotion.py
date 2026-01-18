import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./output"
LABEL_MAP_PATH = MODEL_DIR + "/label_mapping.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer, model, and label mapping
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)


# Helper to predict single or batch texts
def predict(texts, max_length=128, top_k=1):
    # texts: single str or list[str]
    single = isinstance(texts, str)
    if single:
        texts = [texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for prob_vec in probs:
        top_indices = prob_vec.argsort()[::-1][:top_k]
        preds = [
            {
                "label": label_map[str(int(i))],
                "index": int(i),
                "score": float(prob_vec[i]),
            }
            for i in top_indices
        ]
        results.append(preds if top_k > 1 else preds[0])

    return results[0] if single else results


if __name__ == "__main__":
    while True:
        text = input("Enter text (or 'quit'): ").strip()
        if text.lower() in ("quit", "exit"):
            break
        pred = predict(text, top_k=3)
        print(pred)
