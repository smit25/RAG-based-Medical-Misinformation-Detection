from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import nltk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClaimExtractor:
    def __init__(self, claim_model_path = "Nithiwat/deberta-v3-base_claimbuster"): # Load fine-tuned model if available else use defauly model
        self.claim_detector = AutoModelForSequenceClassification.from_pretrained(claim_model_path)  # Claim detection model
        self.claim_detector.to(device)
        self.claim_detector.eval()
        self.claim_tokenizer = AutoTokenizer.from_pretrained(claim_model_path)


    def filter_relevant_sentences(self, transcript):
        relevant_sentences = []
        non_relevant_sentences = []
        sentences = nltk.sent_tokenize(transcript)

        for sentence in sentences:
            inputs = self.claim_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = self.claim_detector(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()

                if predicted_class == 2:
                    relevant_sentences.append(sentence)
                else:
                    non_relevant_sentences.append(sentence)

        return ' '.join(relevant_sentences), ' '.join(non_relevant_sentences)
