import asyncio
from crawl4ai import *
import chromadb
import pprint
import time
import argparse

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Document

from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore

class KnowledgeBaseCreator():
    def __init__(self, db_path, collection_name):
        self.db_path = db_path
        self.db = chromadb.PersistentClient(path=self.db_path)
        self.chroma_collection = self.db.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.hf_embeddings = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        self.llm = Ollama(model="test")
        self.setup_transformations(self.llm)

        try:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store, storage_context=self.storage_context, embed_model=self.hf_embeddings)
        except:
            print("No index found. Creating new index")
            self.index = None

    def setup_transformations(self, llm):
        self.llm_transformations = llm
        self.text_splitter = SentenceSplitter(
            separator=" ",
            chunk_size=1024,
            chunk_overlap=128,
            # paragraph_separator="\n\n\n",
        )
        self.qa_extractor = QuestionsAnsweredExtractor(llm=llm, questions=3)
        self.title_extractor = TitleExtractor(llm=llm, nodes=5)

    async def create_index_from_web_runner(self, url):
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
            )
            # print(result.markdown)
            documents = [
                result.markdown
            ]
            docs = [Document(text=doc) for doc in documents ]
            # print(docs)
            #
            pipeline = IngestionPipeline(
                transformations=[
                    self.text_splitter,
                    self.title_extractor,
                    self.qa_extractor,

                ]
            )
            nodes = await pipeline.arun(
                documents=docs,
                in_place=True,
                show_progress=True
            )
            # pprint.pprint(nodes[0].__dict__)
            # print(len(nodes))

            # index = VectorStoreIndex(nodes, embed_model=hf_embeddings)
            # index.storage_context.persist(persist_dir="./test")
            if not self.index:
                print("Creating new index")
                self.index = VectorStoreIndex([], storage_context=self.storage_context, embed_model=self.hf_embeddings)

            # TODO: Check to see if node exists with the same title or url or other identifier
            print("Inserting nodes")
            self.index.insert_nodes(nodes)

            self.vector_store.persist(persist_path=self.db_path)

    def create_index_from_web(self, url):
        asyncio.run(self.create_index_from_web_runner(url))


class KnowledgeBase():
    def __init__(self, db_path, collection_name):
        self.db_path = db_path
        self.db = chromadb.PersistentClient(path=self.db_path)
        self.chroma_collection = self.db.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.llm = Ollama(model="test")
        self.llm_querying = self.llm
        self.hf_embeddings = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        # self.hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.index = VectorStoreIndex.from_vector_store(self.vector_store, storage_context=self.storage_context, embed_model=self.hf_embeddings)

    def query(self, query):
        start_time = time.time()
        qa = self.index.as_query_engine(llm=self.llm_querying, similarity_top_k=2)
        response = qa.query(query)
        print(f"Time taken: {time.time() - start_time}")
        print(response)
        print(len(response.source_nodes))
        # pprint.pprint(response.source_nodes)
        # pprint.pprint(response.__dict__)



# def query_index():
#     # print("Database done loading")
#     storage_context = StorageContext.from_defaults(persist_dir="./test")
#     index = load_index_from_storage(storage_context=storage_context, embed_model=hf_embeddings)
#
#     start_time = time.time()
#     qa = index.as_query_engine(llm=llm_querying)
#     response = qa.query("What is the major news today?")
#     print(f"Time taken: {time.time() - start_time}")
#     print(response)
#     # pprint.pprint(response.source_nodes)
    # pprint.pprint(response.__dict__)

def main():
    parser = argparse.ArgumentParser(description='Create a knowledge base')
    subparsers = parser.add_subparsers(dest='command')

    create_parser = subparsers.add_parser('create', help='Create a knowledge base')
    create_parser.add_argument('db_path', type=str, help='Path to the database')
    create_parser.add_argument('collection_name', type=str, help='Name of the collection')
    create_parser.add_argument('url', type=str, help='URL to crawl')

    query_parser = subparsers.add_parser('query', help='Query the knowledge base')
    query_parser.add_argument('db_path', type=str, help='Path to the database')
    query_parser.add_argument('collection_name', type=str, help='Name of the collection')
    query_parser.add_argument('query', type=str, help='Query to run')

    args = parser.parse_args()

    if args.command == "create":
        kb = KnowledgeBaseCreator(args.db_path, args.collection_name)
        kb.create_index_from_web(args.url)
        print(kb.chroma_collection.count())
    elif args.command == "query":
        kb = KnowledgeBase(args.db_path, args.collection_name)
        kb.query(args.query)


if __name__ == "__main__":
    main()
