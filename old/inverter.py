import nltk
import re
import spacy
from nltk.corpus import wordnet
from transformers import pipeline



class Inverter:
    def __init__(self):
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

        self.nlp = spacy.load("en_core_web_sm")

    def get_wordnet_pos(self, spacy_token):
        if spacy_token.pos_ == "ADJ":
            return wordnet.ADJ
        elif spacy_token.pos_ == "VERB":
            return wordnet.VERB
        elif spacy_token.pos_ == "ADV":
            return wordnet.ADV
        else:
            return None

    def invert_word(self, word, pos):
        synsets = wordnet.synsets(word, pos=pos)
        if not synsets:
            return word
        for syn in synsets:
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    return lemma.antonyms()[0].name().replace('_', ' ')
        return word

    def invert_sentence(self, sentence: str) -> str:
        doc = self.nlp(sentence)
        new_tokens = []

        for token in doc:
            token_text = token.text

            wn_pos = self.get_wordnet_pos(token)
            replaced = False

            if wn_pos:
                antonym = self.invert_word(token.text.lower(), wn_pos)
                if antonym:
                    if token.text[0].isupper():
                        antonym = antonym.capitalize()
                    new_tokens.append(antonym)
                    replaced = True
                else:
                    if not any(child.dep_ == "neg" for child in token.children):
                        new_tokens.append("not")
                        new_tokens.append(token.text)
                        replaced = True

            if not replaced:
                new_tokens.append(token_text)

        inverted_sentence = " ".join(new_tokens)
        inverted_sentence = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', inverted_sentence)

        return inverted_sentence

    def invert_sentence_using_llm(self, sentence, model_name = "google/flan-t5-small"):
        inverter = pipeline("text2text-generation", model=model_name)
        prompt = (
            f"Rewrite the following sentence so that it expresses the opposite meaning. "
            f"Ensure that the rewritten sentence is grammatically correct, clear, and maintains the original context as much as possible.\n\n"
            f"Original Sentence: \"{sentence}\"\n\n"
            f"Inverted Sentence:"
        )
        results = inverter(prompt, max_length=100, num_return_sequences=1)
        inverted_sentence = results[0]['generated_text'].strip()

        return inverted_sentence

