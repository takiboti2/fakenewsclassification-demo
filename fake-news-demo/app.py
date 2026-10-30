# app.py

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict(headline):
    if not headline.strip():
        return {"Real": 0.0, "Fake": 0.0}

    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    return {
        model.config.id2label[i]: float(probabilities[i])
        for i in range(len(probabilities))
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="News headline",
        placeholder="Enter a news headline...",
        lines=2,
    ),
    outputs=gr.Label(label="Model prediction"),
    title="Fake News Detection Using Transformers",
    description=(
        "Headline-based fake news classification using a fine-tuned "
        "Transformer model. Predictions reflect patterns learned from "
        "the training dataset and are not independent fact-checks."
    ),
    examples=[
        ["Scientists announce a new discovery in renewable energy"],
        ["Government announces changes to national education policy"],
    ],
)

if __name__ == "__main__":
    demo.launch()