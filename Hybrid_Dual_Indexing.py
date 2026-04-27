from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pyspark.sql import SparkSession
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
import contractions
import unicodedata
import joblib
import re


class Keyword_Search:

    def __init__ (self):
        
        self.flatten_chunks = None
        self.bm25 = None
        self.lemmatizer = WordNetLemmatizer ()

    def tokenize (self, text):
        
        text = text.lower()
        text = unicodedata.normalize ('NFKD', text)
        text = contractions.fix (text)

        pattern = r"\w+(?:['-_]\w+)*|[^\w\s]+"
        text = re.findall (pattern, text)
        text = [self.lemmatizer.lemmatize (token) for token in text]

        return text

    def flatten_doc_chunks (self, chunks):

        self.flatten_chunks = []
        for parent_chunk in chunks:
            for child_chunk in parent_chunk:
                self.flatten_chunks.append (child_chunk)

    def BM25 (self, chunks):
        
        self.flatten_doc_chunks (chunks)
        flatten_content_chunks = []

        for chunk in self.flatten_chunks:
            chunk_content = chunk.page_content
            chunk_content_tokenized = self.tokenize (chunk_content)
            flatten_content_chunks.append (chunk_content_tokenized)
        
        self.bm25 = BM25Okapi (flatten_content_chunks)

    def search (self, user_query, top_k = 50):

        user_query_tokenized = self.tokenize (user_query)
        top_k_chunks = self.bm25.get_top_n (user_query_tokenized, self.flatten_chunks, n = top_k)

        return top_k_chunks


class Semantic_Search:
    
    def __init__ (self):
        
        self.embeddings_model = HuggingFaceEmbeddings (model_name = "all-MiniLM-L6-v2")
        self.semantic_chunker = SemanticChunker (self.embeddings_model)
        self.window_chunker = RecursiveCharacterTextSplitter (chunk_size = 1200,
                                                        chunk_overlap = 300,
                                                        separators = ["\n\n", "\n", ".", ". ", " .", ", ", " ,", " ", "  "])

    def overlap_window_splitting (self, parent_chunk, metadata):

        index = 0
        max_word = 200

        while index < len (parent_chunk):
            child_chunk = parent_chunk[index]
            text = child_chunk.page_content
            text_list = re.split (r' ', text)

            if len (text_list) > max_word:
                child_chunk_splitted_by_window = self.window_chunker.split_text (text)

                left_part = parent_chunk[ : index]
                mid_part = []
                right_part = parent_chunk[index + 1 : ]
                
                for child_chunk_text in child_chunk_splitted_by_window:
                    doc = Document (page_content = child_chunk_text, metadata = metadata.copy ())
                    mid_part.append (doc)
                
                parent_chunk = left_part + mid_part + right_part
                index += len (mid_part)
                
            else:
                index += 1

        return parent_chunk

    def semantic_splitting (self, flatten_text, metadata):

        parent_chunk = self.semantic_chunker.create_documents ([flatten_text], metadatas = [metadata])                 
        return parent_chunk

    def chunking (self, dataset):

        chunks = []
        for row in dataset.toLocalIterator ():

            flatten_text = row["flatten_dataset"]
            metadata = {"disease_name": row["json_dataset"]["disease_name"],
                        "source": "SympScan"}
            
            parent_chunk = self.semantic_splitting (flatten_text, metadata)
            parent_chunk = self.overlap_window_splitting (parent_chunk, metadata)

            chunks.append (parent_chunk)
        
        return chunks

    def get_vector_embedding (self, text):

        vector_embedding = self.embeddings_model.embed_query (text)
        return vector_embedding


if __name__ == '__main__':

    spark = SparkSession.builder.appName ('Parquet').getOrCreate ()
    dataset = spark.read.parquet ("Processed_Dataset.parquet")
    
    semantic_search_model = Semantic_Search ()
    keyword_search_model = Keyword_Search ()

    chunks = semantic_search_model.chunking (dataset)
    keyword_search_model.BM25 (chunks)

    joblib.dump (chunks, "Chunks.pkl")
    joblib.dump (semantic_search_model, "Semantic_Model.pkl")
    joblib.dump (keyword_search_model, "Keyword_Model.pkl")