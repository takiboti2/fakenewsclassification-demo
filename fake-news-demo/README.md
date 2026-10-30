# Fake News Classification

A Gradio demo for a DistilBERT model fine-tuned to classify news headlines as real or fake using the WELFake dataset. **Only headlines were used for training and classification; article bodies were excluded.** Predictions reflect learned dataset patterns and do not constitute independent fact-checking.

Run from this directory with the model and tokenizer files in `model/`:

```bash
pip install -r requirements.txt
python app.py
```
