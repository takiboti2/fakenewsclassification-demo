import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent / "model"

TEMPERATURE = 1.4

EXAMPLES = [
    ["Scientists announce a new discovery in renewable energy"],
    ["Government announces changes to national education policy"],
    ["Insider Leaks Bill’s 2-Word Nickname For Hillary, Exposes Dirty Sex Habits"],
    ["U.N. sanctions on North Korea currently in force"],
    ["The Media Go Insane Over Trump In Every Way. That’s Why We Don’t Trust Them."],
    ["Russia WW3 Weapon: Nikola Tesla’s Death Ray In Vladimir Putin’s Possession?"],
    ["At Debate, Hillary Clinton Leaves Questions About Approach to Banks"],
    ["Senate votes to proceed with confirmation vote on SEC nominee"],
    ["U.S. says open to talks on protected status for Haitians"],
]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

print(model.config.id2label)
print(model.config.num_labels)
print(model.config._name_or_path)

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
        logits = model(**inputs, ).logits
        probabilities = torch.softmax(logits /TEMPERATURE, dim=-1)[0]

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
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch()