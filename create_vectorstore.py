import json
import pickle
from typing import List, Dict, Any
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy



class MedicalDataProcessor:
    def __init__(self, data_dir: str = "medical_data", output_dir: str = "medical_db",chunk_size: int = 800,chunk_overlap: int = 50,embedding_model_name: str = "NeuML/pubmedbert-base-embeddings"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        
        # Initialize the text splitter 
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". "]
        )

        # self.text_splitter = TokenTextSplitter(
        #     chunk_size=256,      # ≈ 180–220 words
        #     chunk_overlap=64,    # ≈ 40–50 words overlap
        # )
        
        # Initialize the embedding model
        self.embeddings =  HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")


    def load_json_files(self) -> List[Dict[str, Any]]:
        data = []
        json_files = list(self.data_dir.glob('**/*.json'))
        
        for json_file in json_files:
            try:
                if "nih" in str(json_file):
                    print("found nih file")
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        json_data = json.loads(content)
                        processed_docs = self.process_nih_json_entry(json_data)
                        data.extend(processed_docs)
                else:
                    # For other JSON files, load as is
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        json_data = json.loads(content)
                        
                        if isinstance(json_data, list):
                            data.extend(json_data)
                        else:
                            data.append(json_data)
            except Exception as e:
                print(f"Error loading JSON file {json_file}: {e}")
                continue
        
        print(f"Loaded {len(data)} JSON entries in total")
        return data

    def process_nih_json_entry(self, entry: Dict[str, Any]) -> List[Document]:
        documents = []
        try:
            if 'home_text' in entry:
                for item in entry['home_text']:
                    if 'name' in item:
                        metadata = {
                            'source_type': 'json',
                            'name': 'nih_home_text'
                        }
                        doc = Document(
                            page_content=item['name'],
                            metadata=metadata,
                        )
                        documents.append(doc)
                        
            if 'home_links' in entry:
                for item in entry['home_links']:
                    name = item.get('name', 'Unknown')
                    url = item.get('url', '')
                    link_data = item.get('link_data', '')
            
                    if link_data:
                        sections = self._identify_sections(link_data)
                        
                        for section_title, section_content in sections.items():
                            metadata = {
                                'name': name,
                                'url': url,
                                'source_type': 'json',
                                'section_title': section_title
                            }
                            
                            doc = Document(
                                page_content=section_content,
                                metadata=metadata
                            )
                            documents.append(doc)
                    else:    
                        # If no sections were identified or sections is empty, create a single document
                        if not documents:
                            metadata = {
                                'name': name,
                                'url': url,
                                'source_type': 'json'
                            }
                            doc = Document(page_content=link_data, metadata=metadata)
                            documents.append(doc)
                
        except Exception as e:
            print(f"Error processing JSON entry: {e}")

        print(f"Processed {len(documents)} documents from JSON entry")
            
        return documents


    def _identify_sections(self, text: str) -> Dict[str, str]:
        """
        Identify sections in the text based on common patterns.
        """
     
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            # Check if this line is likely a header
            is_likely_header = (
                line.endswith('?') or 
                (len(line) < 60 and not any(p in line for p in ['.', ',', ';', ':'])) or
                line.startswith('What is') or
                line.startswith('How')
            )
            
            # Check if the next line is blank or very short (potential header indicator)
            next_line_indicator = False
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if not next_line or len(next_line) < 20:
                    next_line_indicator = True
            
            if is_likely_header and (i == 0 or next_line_indicator):
                # Save the previous section before starting a new one
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                    current_content = []
                
                current_section = line
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
            
        return sections


    def load_pdf_files(self) -> List[Document]:
       
        pdf_docs = []
        pdf_files = list(self.data_dir.glob('**/*.pdf'))
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()

                # Add source type metadata
                for doc in documents:
                    doc.metadata['source_type'] = 'pdf'
                    doc.metadata['name'] = pdf_file.name
                
                pdf_docs.extend(documents)
            except Exception as e:
                print(f"Error processing PDF file {pdf_file}: {e}")
                continue
                
        print(f"Loaded {len(pdf_docs)} pages from PDF files")
        return pdf_docs
    

    def load_text_files(self) -> List[Document]:
       
        text_docs = []
        text_files = list(self.data_dir.glob('**/*.txt'))
        print(f"Found {len(text_files)} text files")
        
        for text_file in text_files:
            try:
                loader = TextLoader(str(text_file))
                documents = loader.load()
                
                # Add source type metadata
                for doc in documents:
                    doc.metadata['source_type'] = 'text'
                    doc.metadata['name'] = text_file.name
                
                text_docs.extend(documents)
            except Exception as e:
                print(f"Error processing text file {text_file}: {e}")
                continue
                
        print(f"Loaded {len(text_docs)} text files")
        return text_docs
    

    def load_other_files(self) -> List[Document]:
        other_docs = []
        extensions = [".epub", ".docx", ".doc", ".rtf", ".md"]
        other_files = []
        
        for ext in extensions:
            other_files.extend(list(self.data_dir.glob(f'**/*{ext}')))
        
        print(f"Found {len(other_files)} other document files")
        
        for file_path in other_files:
            try:
                loader = UnstructuredFileLoader(str(file_path))
                documents = loader.load()
                
                # Add source type metadata
                for doc in documents:
                    doc.metadata.clear()
                    doc.metadata['source_type'] = file_path.suffix[1:]  # Remove the dot from extension
                    doc.metadata['name'] = file_path.name
                
                other_docs.extend(documents)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
                
        print(f"Loaded {len(other_docs)} other document files")
        return other_docs
    

    def create_documents(self) -> List[Document]:
      
        all_documents = []
        
        # Process JSON files
        json_docs = self.load_json_files()
        all_documents.extend(json_docs)
        
        # Process PDF files
        # pdf_docs = self.load_pdf_files()
        # all_documents.extend(pdf_docs)
        
        # Process text files
        text_docs = self.load_text_files()
        all_documents.extend(text_docs)
        
        # Process other files
        other_docs = self.load_other_files()
        all_documents.extend(other_docs)
            
        print(f"Created {len(all_documents)} documents from all sources")
        
        all_chunks = []
        for idx, doc in enumerate(all_documents):
            chunks = self.text_splitter.split_documents([doc])
            for idx, chunk in enumerate(chunks):
                chunk.metadata["index"] = idx
                all_chunks.append(chunk)
        # split_documents = self.text_splitter.split_documents(all_documents)
        print(f"Split into {len(all_chunks)} chunks")
        
        return all_chunks


    def build_faiss_database(self, documents: List[Document]) -> FAISS:
        print("Building FAISS database...")

        def relevance_fn(raw_score: float) -> float:
            return max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
        
        faiss_db = FAISS.from_documents(documents, self.embeddings, distance_strategy=DistanceStrategy.COSINE, normalize_L2=True,relevance_score_fn=relevance_fn)

        # Save the FAISS database
        faiss_path = self.output_dir / "faiss_index"
        faiss_db.save_local(str(faiss_path))
        print(f"FAISS database saved to {faiss_path}")
        
        return faiss_db
    
    def build_bm25_database(self, documents: List[Document]) -> BM25Retriever:
        print("Building BM25 retriever...")
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10
        # bm25_retriever.search_type = "similarity"
        
        bm25_path = self.output_dir / "bm25_retriever.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        print(f"BM25 retriever saved to {bm25_path}")
        
        return bm25_retriever
        

    def save_documents(self, documents: List[Document]) -> None:
        docs_path = self.output_dir / "processed_documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(documents, f)
        print(f"Saved {len(documents)} processed documents to {docs_path}")
        

    def load_documents(self, file_path: str) -> List[Document]:
        with open(file_path, 'rb') as f:
            documents = pickle.load(f)
        return documents
    

    def run(self) -> Dict[str, Any]:
        # Create documents
        documents = self.create_documents()
        
        # Save the processed documents
        self.save_documents(documents)
        
        # Build FAISS database
        faiss_db = self.build_faiss_database(documents)
        
        # Build BM25 retriever
        bm25_retriever = self.build_bm25_database(documents)
        
        return {"faiss_db": faiss_db, "bm25_retriever": bm25_retriever, "document_count": len(documents)}
    


if __name__ == "__main__":
    data_dir = "medical_data"
    db_dir = "medical_db"

    processor = MedicalDataProcessor(
        data_dir=data_dir,
        output_dir=db_dir,
        chunk_size=1000,
        chunk_overlap=100
    )
    
    result = processor.run()
    print(f"Processing complete! Created {result['document_count']} documents.")
    
    

    