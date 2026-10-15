from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

if __name__ == "__main__":
    print("Ingesting...\n")

    loader = TextLoader("./test.txt", encoding="utf-8")
    document = loader.load()

    print("Splitting...\n")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)
    print(f"Created {len(chunks)} chunks\n")

    print("Embedding...\n")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=SecretStr(settings.OPENAI_API_KEY)
    )

    print("Ingesting  ...\n")
    PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=settings.PINECONE_INDEX_NAME
    )

    print("Done!\n")
