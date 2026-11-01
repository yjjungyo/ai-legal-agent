import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY","")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT","")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION","2024-10-21")

AZURE_OPENAI_DEPLOYMENT_CHAT = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT","aicore-gpt4o")
AZURE_OPENAI_DEPLOYMENT_EMBED = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED","text-embedding-3-large")

CHROMA_PERSIST = os.getenv("CHROMA_PERSIST", "./vectorstore/chroma")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./storage/app.db")
DATA_PATH = os.getenv("DATA_PATH", "./data")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_PATH  = os.getenv("LOG_PATH", "./storage/app.log")

LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
LANGFUSE_CAPTURE_CONTENT = os.getenv("LANGFUSE_CAPTURE_CONTENT", "false").strip().lower() in {"1", "true", "yes", "on"}
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_SYSTEM_NAME = os.getenv("LANGFUSE_SYSTEM_NAME", "LegalAgent")

def get_llm()->AzureChatOpenAI:
    """Azure OpenAI Chat 인스턴스 생성"""
    return AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_CHAT,
        temperature=0,
        stream_usage=True,
        name=LANGFUSE_SYSTEM_NAME,
        tags=[LANGFUSE_SYSTEM_NAME],
    )


# ---------------------------
# Embeddings / LLM Builders
# ---------------------------

def get_embeddings() -> AzureOpenAIEmbeddings:
    """Azure OpenAI Embeddings 인스턴스 생성"""
    return AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_EMBED,
    )


def get_temp_llm(temperature: float = 0) -> AzureChatOpenAI:
    """Azure Chat LLM 인스턴스 생성"""
    return AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_CHAT,
        temperature=temperature,
        stream_usage=True,
        name=LANGFUSE_SYSTEM_NAME,
        tags=[LANGFUSE_SYSTEM_NAME],
    )

