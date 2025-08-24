import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load a pre-trained transformer model (fine-tuning recommended for production)
MODEL_NAME = "roberta-base"  # You can use more advanced models if needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)  # 0: Neutral, 1: Ageism, 2: Sexism, 3: Racial Bias

# Example bias types
BIAS_LABELS = ["Neutral", "Ageism", "Sexism", "Racial Bias"]

# Placeholder for inclusive suggestions (expand for production)
INCLUSIVE_SUGGESTIONS = {
    "Ageism": "Consider using language that values all age groups.",
    "Sexism": "Use gender-neutral terms and avoid stereotypes.",
    "Racial Bias": "Ensure language is respectful and inclusive of all races.",
    "Neutral": "No bias detected."
}

def classify_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(scores, dim=1).item()
        bias_type = BIAS_LABELS[pred_label]
        suggestion = INCLUSIVE_SUGGESTIONS[bias_type]
    return {"bias_type": bias_type, "suggestion": suggestion, "confidence": scores[0][pred_label].item()}

# Example usage
if __name__ == "__main__":
    sample_text = "The young team lacks experience compared to older employees."
    result = classify_bias(sample_text)
    print(f"Bias Type: {result['bias_type']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Suggestion: {result['suggestion']}")
