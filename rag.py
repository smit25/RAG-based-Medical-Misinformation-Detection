from typing import List, Dict, Any, Tuple, Optional, TypedDict
import os
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores.utils import DistanceStrategy
from langgraph.graph import StateGraph
from langgraph.graph.graph import END
import re
import numpy as np
import nltk
import json

nltk.download("punkt", quiet=True)

# from ranker import Ranker
# from inverter import Inverter
from rus_ranker import RUSRanker
from trustworthiness import (
    compute_trust_and_uncertainty,
    beta_posterior_trust
)

from dotenv import load_dotenv
load_dotenv()

# Define our state
class VerificationState(TypedDict):
    transcript_chunk: str
    evidence: str
    correct_claims: List[str]
    unverifiable_claims: List[str]
    potential_misinformation: List[str]
    verification_details: List[Dict[str, Any]]
    claim_labels: List[str]


class MisinformationDetector:
    """
    RAG bot for detecting misinformation in diabetes content.
    """
    
    def __init__(
        self,
        temperature: float = 0.0,
        top_k: int = 5
    ):
        """
        Initialize the detector.
        """

        self.embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

        self.faiss_store = self.load_faiss_index("medical_db/faiss_index")
        self.bm25_retriever = self.load_bm25_retriever("medical_db/bm25_retriever.pkl")
        # inverter = Inverter()
        
        self.ranker = RUSRanker(
            faiss_store=self.faiss_store,
            bm25_retriever=self.bm25_retriever,
            # inverter=inverter,
            rank_type="irc"
        )
        
        # Initialize LLM
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=temperature,
            max_tokens=1000,
            timeout=None,
            max_retries=2,
            google_api_key=gemini_api_key
            # streaming=True  # Enable streaming

        )
        
        self.top_k = top_k

        self.ce = CrossEncoder("ivan-savchuk/cross-encoder-ms-marco-MiniLM-L-12-v2-tuned_mediqa-v1")

        self.bm25_weight = 0.3


    def load_faiss_index(self, faiss_path: str) -> FAISS:
        """
        Load a pre-built FAISS index.
        """

        def clamp_cosine(raw_score: float) -> float:
            # map FAISS raw cosine [-1,1] → [0,1]
            return max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
        
        return FAISS.load_local(faiss_path, 
                                self.embeddings, 
                                allow_dangerous_deserialization=True,
                                distance_strategy=DistanceStrategy.COSINE,
                                relevance_score_fn=clamp_cosine,
                                normalize_L2=True,
                                )


    def load_bm25_retriever(self, bm25_path: str) -> BM25Retriever:
        """
        Load a pre-built BM25 retriever.
        """
        with open(bm25_path, 'rb') as f:
            return pickle.load(f)

    
    def format_context(self, documents: List[Document]) -> str:
        context_parts = []
        
        for i, doc in enumerate(documents):
            source_id = doc.metadata.get("id", f"document_{i}")
            rus_impact = doc.metadata.get("rus_impact", "N/A")
            context_parts.append(f"Document {i+1} [ID: {source_id}, RUS Impact: {rus_impact}]:\n{doc.page_content}\n")
            
        return "\n".join(context_parts)
    
    def finalize_trust(self, state: Dict[str, Any]) -> Dict[str, Any]:
       
        trustworthiness_score, uncertainty, lb, ub = compute_trust_and_uncertainty(
            n_correct=corr_len,
            n_unverif=unv_len,
            n_wrong=mis_len,
            severities=[ver["severity_score"] for ver in ver_dets],
            confidences_unverif=result.get("unverifiable_confidences", []),
            alpha=0.5,
            beta=0.8,
            lambda_uncertainty=0.5
        )

        return trustworthiness_score, uncertainty, lb, ub
    

    def detect_contradictions(self, state: VerificationState) -> VerificationState:
        detection_prompt = PromptTemplate(
            input_variables=["transcript_chunk", "evidence"],
            template="""
            You are a medical fact–checker. Compare the TRANSCRIPT against the MEDICAL EVIDENCE, using only that evidence.

            TRANSCRIPT:
            "{transcript_chunk}"

            MEDICAL EVIDENCE:
            {evidence}

            — TASK —
            1. Extract every concrete factual claim about diabetes.
            • Tag each claim [ACCURATE], [INACCURATE], or [UNVERIFIABLE] based strictly on the evidence.
            • If a claim refers to a general condition (e.g., “the disease”), and the evidence describes a specific subtype (e.g., “Type A of that disease”) with the same defining characteristic, treat the subtype description as SUPPORT for the general claim.
            • Number them 1., 2., 3., …

            2. Under POTENTIAL MISINFORMATION, list all claims tagged [INACCURATE]. If none, write “None.”

            — OUTPUT —
            CLAIMS:
            1. <claim> — [ACCURATE/INACCURATE/UNVERIFIABLE]
            2. ...
            n. ...

            POTENTIAL MISINFORMATION:
            <each INACCURATE claim on its own line, or “None”>
            """
        )
        result = self.llm.invoke(
            detection_prompt.format(
                transcript_chunk=state["transcript_chunk"],
                evidence=state["evidence"]
            )
        )

        raw_txt = result.content.strip()
        claims_block = re.search(r"CLAIMS:(.*?)(?:POTENTIAL MISINFORMATION:)", raw_txt, re.S)
        claims_lines = claims_block.group(1).strip().splitlines() if claims_block else []
        potential_block = re.search(r"POTENTIAL MISINFORMATION:(.*)$", raw_txt, re.S)
        inaccurate_claims = [
            ln.strip().strip('" ')
            for ln in potential_block.group(1).splitlines()
            if ln.strip() and ln.strip().lower() != "none"
        ] if potential_block else []
        correct_claims = []
        unverifiable_claims = []

        claim_labels = []
        for line in claims_lines:
            m = re.match(r"\s*\d+\.\s*(.*?)\s*—\s*\[(ACCURATE|INACCURATE|UNVERIFIABLE)\]", line)
            if not m:
                continue
            claim_text, flag = m.groups()
            claim_text = claim_text.strip().strip('" ').strip('<>').strip()
            if flag == "UNVERIFIABLE":
                unverifiable_claims.append(claim_text)
            elif flag == "ACCURATE":
                correct_claims.append(claim_text)

            claim_labels.append(flag)
        
        return {
            **state,
            "claim_labels": claim_labels,
            "correct_claims": correct_claims,
            "unverifiable_claims": unverifiable_claims,
            "potential_misinformation": inaccurate_claims
        }


    def verify_misinformation(self, state: VerificationState) -> VerificationState:
        verification_prompt = PromptTemplate(
            input_variables=["transcript_chunk", "evidence", "potential_misinformation"],
            template="""
            Let’s think this through carefully:

            — REFLECTION —
            1. Have I compared the claim fully to the evidence?
            2. Is my severity assessment justified?

            Now verify the following claim:

            TRANSCRIPT:
            "{transcript_chunk}"

            MEDICAL EVIDENCE:
            {evidence}

            CLAIM TO VERIFY:
            "{potential_misinformation}"

           — TASK (per claim) —
                1. Quote the claim.
                2. Find the exact sentence(s) in the evidence that address it.
                3. **CORRECTION:** Paraphrase those sentence(s) into a concise statement directly refuting or clarifying the claim.  
                • Do **not** introduce any information not in the evidence.  
                • If the evidence does not address the claim at all, write:  
                    `No correction available based on provided evidence.`
                4. **SEVERITY_SCORE:**  
                    A two-decimal number (0.00–1.00) reflecting  
                    – **Potential harm** (health, safety, legal, financial)  
                    – **Spread likelihood** (how credibly it may be accepted)  
                    

                — OUTPUT (repeat block for each claim) —
                CLAIM: "<exact claim text>"
                CORRECTION: "<paraphrased evidence or 'No correction available based on provided evidence.'>"
                SEVERITY_SCORE: <0.00–1.00>
            """
        )

        potential = [
            c for c in state.get("potential_misinformation", [])
            if c.strip()
        ]
        verification_details = []
        for claim in potential:
            res = self.llm.invoke(
                verification_prompt.format(
                    transcript_chunk=state["transcript_chunk"],
                    evidence=state["evidence"],
                    potential_misinformation=claim
                )
            ).content.strip()


            clm   = re.search(r'CLAIM:\s*"(.*?)"', res, re.S)
            cor = re.search(r'CORRECTION:\s*(?:"(.*?)"|(.+))', res)
            score = re.search(r'SEVERITY_SCORE:\s*\*{0,2}\s*([01]\.\d{2})', res, re.S)

            if score:
                sev_score = float(score.group(1))
                if   sev_score <= 0.33: sev_label = "Minor"
                elif sev_score <= 0.66: sev_label = "Moderate"
                else:                  sev_label = "Severe"
            else:
                sev_score = None
                sev_label = None

            verification_details.append({
                "claim":          clm.group(1) if clm else claim,
                "correction":     (cor.group(1) or cor.group(2) or "").strip(),
                "severity":        sev_label,
                "severity_score": sev_score
            })

        return {
            **state,
            "verification_details": verification_details
        }


    def should_verify_misinformation(self, state: VerificationState) -> str:
        return "verify" if state.get("potential_misinformation") and state.get("potential_misinformation") != [] else "end"
    

    def create_verification_graph(self):
        graph = StateGraph(VerificationState)
        graph.add_node("contradiction_detection", self.detect_contradictions)
        graph.add_node("misinformation_verification", self.verify_misinformation)

        graph.add_conditional_edges(
            "contradiction_detection",
            self.should_verify_misinformation,
            {
                "verify": "misinformation_verification", 
                "end": END
            }
        )
        graph.set_entry_point("contradiction_detection")
        return graph.compile()

    
    def score_unverifiable_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Key idea here is that the if it is a factual claim not present in the evidence, it is likely misinformation and should be penalized
        
        """

        claims: List[str] = state.get("unverifiable_claims", [])
        if not claims:
            return state

        numbered = "\n".join(f"{i+1}. \"{c}\"" for i, c in enumerate(claims))

        batch_prompt = PromptTemplate(
            input_variables=["claims_list"],
            template="""
            You are a precise fact-evaluator. Below is a numbered list of sentences flagged UNVERIFIABLE.
            Output a JSON array of floating-point confidences between 0 and 1, in the same order as the list,
            indicating how likely each sentence is a factual medical claim requiring evidence.
            Respond with only the JSON array, e.g.: [0.75, 0.20, 0.90].

            {claims_list}
            """
        ).format(claims_list=numbered)

        resp = self.llm.invoke(batch_prompt)
        content = resp.content.strip()

        try:
            confidences = json.loads(content)
            if isinstance(confidences, list) and all(isinstance(x, (int, float)) for x in confidences):
                return {**state, "unverifiable_confidences": [float(x) for x in confidences]}
        except json.JSONDecodeError:
            pass

        nums = re.findall(r"0(?:\.\d+)?|1(?:\.0+)?", content)
        confidences = [float(n) for n in nums[: len(claims)]]

        return {**state, "unverifiable_confidences": confidences}

        
    def get_relevance_scores(self, query, docs, use_bm25 = False) -> List[float]:
        scores = self.ce.predict([(query, doc.page_content) for doc in docs])

        max_score = max(scores)
        min_score = min(scores)
        range_score = max_score - min_score if max_score > min_score else 1
        scores = [(s - min_score) / range_score for s in scores]

        if use_bm25:
            tokens = self.bm25_retriever.preprocess_func(query)
            all_bm25 = self.bm25_retriever.vectorizer.get_scores(tokens)
            # Map each doc.name -> its BM25 score (zero if missing)
            name2bm25 = {
                doc.metadata["name"]: sc
                for doc, sc in zip(self.bm25_retriever.docs, all_bm25)
            }
            bm25_scores = [ name2bm25.get(d.metadata["name"], 0.0) for d in docs ]
            
            lo_bm, hi_bm = min(bm25_scores), max(bm25_scores)
            bm25_norm = [(s - lo_bm) / (hi_bm - lo_bm if hi_bm > lo_bm else 1) for s in bm25_scores]
            
            final_scores = [
                (1 - self.bm25_weight) * ce + self.bm25_weight * bm25
                for ce, bm25 in zip(scores, bm25_norm)
            ]
    
            for doc, score in zip(docs, final_scores):
                doc.metadata["relevance_score"] = score
        else:
            final_scores = scores
            for doc, score in zip(docs, final_scores):
                doc.metadata["relevance_score"] = score
    

        return docs, final_scores
    

    def detect_misinformation(self, query: str) -> str:

        sentences = nltk.sent_tokenize(query)
        final_docs = []
        for sent in sentences:
            sim_docs, sim_scores = self.ranker.get_documents_by_similarity(sent, 5)
            rel_docs, rel_scores = self.get_relevance_scores(sent, sim_docs)

            documents = self.ranker.rus_search(sim_docs, sim_scores, rel_scores, k=self.top_k)
            docs_list = []
            for doc in documents:
                clean_metadata = {}
                for k, v in doc.metadata.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean_metadata[k] = float(v)
                    else:
                        clean_metadata[k] = v

                docs_list.append({
                    "page_content": doc.page_content,
                    "metadata": clean_metadata
                })

            docs_json = json.dumps(docs_list, indent=4)
            with open("documents.json", "w") as f:
                f.write(docs_json)
            print(len(documents))

            if not documents:
                return {"answer": "No relevant documents found to answer the query.", "documents": []}
            
            for doc in documents:
                final_docs.append(doc.page_content)
        
        # for doc in final_docs:
        #     print(doc)

        evidence_text = "\n".join([chunk for chunk in final_docs])
        
        # Initialize state
        initial_state = {
            "transcript_chunk": query,
            "evidence": evidence_text,
            "correct_claims": None,
            "potential_misinformation": None,
            "unverifiable_claims": None,
            "verification_details": None,
            "claim_labels": None,
        }
        
        graph = self.create_verification_graph()
        final_state = graph.invoke(initial_state)

        return final_state




if __name__ == "__main__":
    detector = MisinformationDetector()

    with open("test_set.json", "r") as f:
        test_set = json.load(f)

    test_results = []

    print(len(test_set))
    
    for test in test_set:
        id = test.get("id", "-1")
        severity_scores = []
        query = test.get("transcript", "")
        category = test.get("category", "general")
        ground_severity = test.get("severity", [])
        ground_unverifiable = test.get("unverifiable_confidence", [])
        ground_trust = test.get("trustworthiness_score", -1.0)
        ground_claims = test.get("claims", [])
        # query = "Diabetes occurs when Pancreas creates excess insulin. Many people with diabetes also develop high blood pressure."
        result = detector.detect_misinformation(query)
        
        corr_len = len(result.get("correct_claims", []))
        unv_len = len(result.get("unverifiable_claims", []))
        mis_len = len(result.get("potential_misinformation", []))
        ver_dets = result.get("verification_details", [])
        if ver_dets is None:
            ver_dets = []

        assert(corr_len + unv_len + mis_len == len(result.get("claims", [])), 
            "Mismatch between claims and correct/unverifiable/misinformation claims length")

        assert(mis_len == len(ver_dets), [], 
            "Mismatch between potential misinformation and verification details length")
        
        if unv_len > 0:
            result = detector.score_unverifiable_confidence(result)
            print(f"Unverifiable Claim Len: {len(result['unverifiable_claims'])}")
            print(f"Unverifiable Confidences: {result['unverifiable_confidences']}")
            print(f"Ground Unverifiable Len: {len(ground_unverifiable)}")
        
        # for i in range(corr_len):
        #     severity_scores.append(0)

        # for i in range(unv_len):
        #     severity_scores.append(0.5)
        # print(ver_dets)
        for i, ver in enumerate(ver_dets):
            severity_scores.append(ver["severity_score"])

        assert(len(severity_scores) == len(ground_severity),
            "Mismatch between severity scores and ground severity scores")
        
        assert(len(result.get("unverifiable_confidences", [])) == unv_len,
            "Mismatch between unverifiable confidences and unverifiable claims length")

        trustworthiness_score, uncertainty, lb, ub = compute_trust_and_uncertainty(
            n_correct=corr_len,
            n_unverif=unv_len,
            n_wrong=mis_len,
            severities=[ver["severity_score"] for ver in ver_dets],
            confidences_unverif=result.get("unverifiable_confidences", []),
            alpha=0.5,
            beta=0.7,
            lambda_uncertainty=0.5
        )

        print("RES:", result.get("claim_labels", []))

        result.update({
            "id":                    id,
            "category":              category,
            "ground_severity_scores": ground_severity,
            "ground_unverifiable":   ground_unverifiable,
            "ground_trustworthiness_score":          ground_trust,
            "ground_claim_labels":    ground_claims,
            "predicted_claim_labels": result.get("claim_labels", []),
            # "claims":                result.get("correct_claims", []) + result.get("unverifiable_claims", []) + result.get("potential_misinformation", []),
            "severity_scores":       severity_scores,
        })

        # Trustworthiness score and Uncertainty to the result
        result.update({
            "trustworthiness_score": trustworthiness_score,
            "uncertainty":            uncertainty,
            "lower_bound":            lb,
            "upper_bound":            ub,
        })
    
        test_results.append(result)

    result_json = json.dumps(test_results, indent=4)
    with open("result_test_set.json", "w") as f:
        f.write(result_json)


    print(f"Trustworthiness Score: {trustworthiness_score:.4f} +- {uncertainty:.4f} ({lb:.4f}, {ub:.4f})")

    # trustworthiness_score, uncertainty, lower_q, upper_q = beta_posterior_trust(
    #     n_correct=corr_len,
    #     n_unverif=unv_len,
    #     n_wrong=mis_len,
    #     severities=[ver["severity_score"] for ver in ver_dets],
    #     alpha_credit=0.5,
    #     gamma=2.0,
    #     alpha0=2.0,
    #     beta0=2.0
    # )

    # print(f"Trustworthiness Score: {trustworthiness_score:.4f} + {uncertainty:.4f} ({lower_q:.4f}, {upper_q:.4f})")
