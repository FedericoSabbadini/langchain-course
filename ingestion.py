import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi # Ensure certifi is imported to access the CA bundle
# vertifi is needed to ensure that the SSL context uses the correct CA certificates, especially in environments where the default certificates may not be up-to-date or properly configured.
from dotenv import load_dotenv
# from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings # embedding
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
# certifi.where() returns the path to the CA bundle provided by certifi, which is a collection of trusted CA certificates. 
# By setting the SSL context to use this CA bundle, we ensure that our application can establish secure HTTPS connections without running into certificate verification issues.
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
# attention if using a VPN or proxy that intercepts SSL traffic, you may need to configure the SSL context to trust the proxy's certificate as well.

EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    #show_progress_bar=False,
    #chunk_size=50, # limit of 50 documents per batch to avoid rate limits
    # id too high, you may encounter rate limits or timeouts when processing large batches of documents. Adjusting the chunk size to a smaller value can help mitigate these issues and ensure smoother processing.
   # retry_min_seconds=10,
)
#vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings) # it is a local vector store that can be used for development and testing. It allows you to store and query vector embeddings without needing an external service. This can be useful for small-scale projects or when you want to avoid the complexity of setting up a separate vector database. However, for larger projects or production use, you may want to consider using a more robust vector store like Pinecone, which is designed for scalability and performance.
vectorstore = PineconeVectorStore( # it is a cloud-based vector database service that provides scalable and efficient storage and querying of vector embeddings. It is designed for production use and can handle large volumes of data with low latency. Using Pinecone can be beneficial for applications that require high performance and scalability, especially when dealing with large datasets or real-time querying. However, it may involve additional costs compared to a local vector store like Chroma.
     index_name="langchain-docs-2025", embedding=embeddings
)
tavily_extract = TavilyExtract() # it is a tool that can be used to extract specific information from web pages during the crawling process. 
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000) # it is a tool that can be used to control the crawling process by defining parameters such as maximum depth, breadth, and number of pages to crawl. This allows you to manage the scope of the crawl and ensure that it stays within reasonable limits, preventing excessive crawling that could lead to performance issues or overwhelming amounts of data. By setting these parameters, you can focus the crawl on the most relevant sections of the documentation while avoiding unnecessary pages.
tavily_crawl = TavilyCrawl() # it is a tool that can be used to perform the actual crawling of the documentation site. 
# It allows you to specify the starting URL and other parameters to control the crawling process. The TavilyCrawl tool will navigate through the specified URL and its linked pages, extracting content based on the defined extraction rules and mapping parameters. This is a crucial component of the documentation ingestion pipeline, as it enables you to gather the raw content that will later be processed and indexed in the vector store.
# Search = ricerca di URL a cui sono dei dati; Extract = estrazione dei dati da un URL; Map = controllo del processo di crawling (profondità, ampiezza, numero di pagine); Crawl = processo di crawling vero e proprio (navigazione e raccolta dei dati o tutto ciò che è associato a una pagina).

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.add_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
            'instructions': "Extract the main content of the page, including text, code snippets, and relevant metadata. Ignore navigation menus, footers, and advertisements. Focus on extracting information that would be useful for understanding the documentation and its structure.",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
