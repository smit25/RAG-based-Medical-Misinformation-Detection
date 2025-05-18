class RAGPipeline:
    def __init__(self, claim_model_path = "Misinformation/models/deberta"):
        self.claim_detector = AutoModelForSequenceClassification.from_pretrained(claim_model_path).to(device)  # Claim detection model
        self.claim_tokenizer = AutoTokenizer.from_pretrained(claim_model_path)
        self.faiss_indexer = FAISSIndex()
        self.fact_check_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
        self.fact_check_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.document_fetcher = DocumentFetcher()
        self.inverter = Inverter()
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2").to(device) 


    def chunk_text_manual(self, text, chunk_size, overlap):
        sentences = nltk.sent_tokenize(text)
        chunks = []

        for i in range(0, len(sentences) - chunk_size + 1, chunk_size - overlap):
            chunk = " ".join(sentences[i:i + chunk_size])
            chunks.append(chunk)

        return chunks


    def filter_relevant_sentences(self, transcript):
        self.claim_detector.to(device)
        self.claim_detector.eval()
        relevant_sentences = []
        sentences = nltk.sent_tokenize(transcript)

        for sentence in sentences:
            inputs = self.claim_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = self.claim_detector(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()

                if predicted_class == 2:
                    relevant_sentences.append(sentence)

        print("Relevant Sentences:", relevant_sentences)
        return ' '.join(relevant_sentences)


    def generate_response_using_mini_check(self, query, documents):
        evidence = " ".join([str(doc) for doc in documents])
        scorer = MiniCheck(model_name='flan-t5-large', cache_dir='Misinformation/ckpts')
        pred_label, raw_prob, _, _ = scorer.score(docs=[evidence], claims=[query])

        return raw_prob[0]


    def generate_response(self, query, documents):
        evidence = " ".join([str(doc) for doc in documents])
        input_text = textwrap.dedent(f"""
        Fact-checking Task:
        Statement: {query}
        Evidence: {evidence}
        Please read the statement and the evidence carefully.
        Then, determine if the statement is supported by the evidence.
        Respond with a single word: 'True' if supported, or 'False' if not.
        """).strip()
        input_ids = self.fact_check_tokenizer.encode(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        outputs = self.fact_check_model.generate(input_ids, max_length=50, num_beams=4)
        result = self.fact_check_tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result.strip().lower()

        if result == 'True':
            return 'True'
        elif result == 'False':
            return 'False'


    def append_to_json(self, new_data_, video_id):
        with open('Misinformation/outputs.json', 'a+') as f:
            json.dump(new_data_, f, indent=4)
            f.write(',\n')
            f.close()

        with open('Misinformation/processed_transcripts_list.json', "a+") as f:
            json.dump(video_id, f)
            f.write("\n")

        return


    def detect_misinformation(self, keyword, transcript):
        results = []
        # filtered_transcript = self.filter_relevant_sentences(transcript)
        filtered_transcript = transcript
        chunks = self.chunk_text_manual(filtered_transcript, 2, 1)

        documents = self.document_fetcher.fetch_all_data(keyword)
        document_chunks = self.chunk_text_manual(documents, 3, 1)

        self.faiss_indexer.indexed_chunks = [{"id": i, "text": doc} for i, doc in enumerate(document_chunks)]
        self.bm25 = BM25Okapi(document_chunks, k1=1.2, b=0.75)

        if documents:
            embeddings = self.faiss_indexer.encode_documents(documents)
            self.faiss_indexer.build_faiss_index(embeddings)
            self.corpus_ids = [chunk["id"] for chunk in self.faiss_indexer.indexed_chunks]
        else:
            logger.error("No documents fetched for the keyword.")
            return results
        self.reranker = Ranker(self.bm25, self.faiss_indexer, self.corpus_ids, self.inverter, "irc")

        torch.cuda.empty_cache()
        chunk_probs = []
        for chunk in chunks:
            retireved_results = self.reranker.rerank(chunk, top_k = 10)

            if not retireved_results:
                results.append([-1]*len(thresholds))
                continue

            final_docs = [document_chunks[int(doc_id)] for doc_id, _ in retireved_results]
            raw_probs = self.generate_response_using_mini_check(chunk, final_docs)
            # if raw_probs < 0.01:
            #     continue
            chunk_probs.append(raw_probs)

        return chunk_probs

