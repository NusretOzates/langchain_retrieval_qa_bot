from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def load_text_file(file_path: str) -> Document:
    """Loads a text file and returns a Document object.

    Args:
        file_path: Path to the text file.

    Returns:
        A Document object.
    """
    doc = TextLoader(file_path, encoding="utf-8").load()[0]
    return doc


def load_pdf_file(file_path: str) -> List[Document]:
    """Loads a pdf file and returns a list of Document objects.

    Args:
        file_path: Path to the pdf file.

    Returns:
        A list of Document objects. Every page in the pdf file is a Document object.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def create_index(docs: List[Document]) -> ContextualCompressionRetriever:
    """Creates a vectorstore index from a list of Document objects.

    Args:
        docs: List of Document objects.

    Returns:
        A vectorstore index. It searches the most similar document to the given query but with
        the help of MMR it also tries to find the most diverse document to the given query.

    """

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    llm = ChatOpenAI()
    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        text_splitter=splitter,
    ).from_documents(docs)

    retriever = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=index.vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "score_threshold": 0.5}
        ),
    )

    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return retriever
