import os
import time
import warnings
import subprocess
from datetime import datetime
from typing import List, Optional, TypedDict, Literal, Type, TypeVar
import json
import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup
from voice_io import listen, speak


from pydantic import BaseModel, Field, field_validator, model_validator

from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.documents import Document

from openai import OpenAI

from guardrails import Guard, OnFailAction
from guardrails.hub import DetectPII, ToxicLanguage
from dotenv import load_dotenv

load_dotenv() 


EPISODIC_MEMORY_PATH = "episodic_memory.json"
PROCEDURAL_RULES_PATH = "procedural_rules.json"
CONVERSATION_MEMORY_PATH = "conversation_memory.json"


DEFAULT_RULES = {
    "career": "career_retrieve_node",
    "announcement": "announcements_retrieve_node",
    "greeting": "direct_node",
    "default": "static_retrieve_node"
}

def load_procedural_rules():
    if not os.path.exists(PROCEDURAL_RULES_PATH):
        with open(PROCEDURAL_RULES_PATH, "w") as f:
            json.dump(DEFAULT_RULES, f, indent=2)
    with open(PROCEDURAL_RULES_PATH) as f:
        return json.load(f)


def load_conversation_memory():
    if not os.path.exists(CONVERSATION_MEMORY_PATH):
        return []
    with open(CONVERSATION_MEMORY_PATH) as f:
        return json.load(f)

def append_conversation(q, a):
    mem = load_conversation_memory()
    mem.append({"q": q, "a": a})
    mem = mem[-50:]
    with open(CONVERSATION_MEMORY_PATH, "w") as f:
        json.dump(mem, f, indent=2)

# =========================================================
# CONFIG (inline)
# =========================================================

class config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # Embeddings
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    # Vector DB (ABSOLUTE PATHS – REQUIRED)
    faiss_path = os.getenv(
        "faiss_path",
        r"C:\Users\afarooq\Desktop\langgraph-bot\faiss_data"
    )

    data_path = os.getenv(
        "data_path",
        r"C:\Users\afarooq\Desktop\langgraph-bot\lmkr_data"
    )

    static_output = os.getenv("static_output", "static.txt")

    # Retrieval
    k_static = int(os.getenv("k_static", "5"))
    k_careers = int(os.getenv("k_careers", "5"))
    k_announcements = int(os.getenv("k_announcements", "5"))


    # Scraping URLs (set yours if different)
    career_url = os.getenv("career_url", "https://lmkr.bamboohr.com/careers")
    announcements_url = os.getenv("announcements_url", "https://lmkr.com/announcements/")

    # Cache files
    save_career_output = os.getenv("save_career_output", "careers_cache.txt")
    save_announcement_output = os.getenv("save_announcement_output", "news_cache.txt")

    scrape_after = int(os.getenv("scrape_after", "6"))
    scrape_timeout = int(os.getenv("scrape_timeout", "20"))

    # Selenium
    scrape_wait_time = float(os.getenv("scrape_wait_time", "4"))
    web_user_agent = os.getenv(
        "web_user_agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )

    # HTML cleanup
    tags_cleanup = ["script", "style", "noscript", "svg"]

    # Guardrails
    PII_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER"]
    TOXIC_LANGUAGE_THRESHOLD = float(os.getenv("TOXIC_LANGUAGE_THRESHOLD", "0.8"))





def fetch_raw_html(url: str) -> str:
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService

    edge_options = EdgeOptions()
    edge_options.add_argument("--headless=new")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument(f"--user-agent={config.web_user_agent}")

    driver = webdriver.Edge(options=edge_options, service=EdgeService())
    try:
        driver.get(url)
        time.sleep(config.scrape_wait_time)
        return driver.page_source
    finally:
        driver.quit()


def extract_bamboohr_job_links_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.select("a[href*='/careers/']"):
        href = a.get("href")
        if href:
            if href.startswith("/"):
                href = "https://lmkr.bamboohr.com" + href
            links.append(href)

    return list(set(links))



# =========================================================
# UTILS (inline)
# =========================================================

def save_to_file(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def is_file_fresh(path: str, hours: int) -> bool:
    if not os.path.exists(path):
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds < (hours * 3600)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120
)

def split_text_into_chunks(text: str) -> List[str]:
    return text_splitter.split_text(text)


def clean_text_content(text: str) -> str:

    lines = []
    for line in text.splitlines():
        line = " ".join(line.strip().split())
        if line:
            lines.append(line)
    return "\n".join(lines)


# =========================================================
# EMBEDDINGS + VECTORSTORE + MEMORY
# =========================================================

if not config.OPENAI_API_KEY:
    print("OPENAI_API_KEY is not set. Export it before running for full functionality.")

embeddings = OpenAIEmbeddings(
    model=config.EMBEDDINGS_MODEL,
    openai_api_key=config.OPENAI_API_KEY
)

memory_store = InMemoryStore(
    index={
        "dims": config.EMBEDDING_DIMENSION,
        "embed": embeddings
    }
)


# =========================================================
# VECTORSTORE BOOTSTRAP (PERSISTENT FAISS)
# =========================================================

# =========================================================
# VECTORSTORE BOOTSTRAP (PERSISTENT FAISS)
# =========================================================

vectorstore = None
documents = []

# 1️⃣ Try loading existing FAISS
if os.path.exists(config.faiss_path):
    try:
        vectorstore = LC_FAISS.load_local(
            config.faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS index loaded from disk.")
    except Exception as e:
        print(f"⚠️ Failed to load FAISS index: {e}")

# 2️⃣ If not found or failed → build & persist
if vectorstore is None:
    print("⚠️ No FAISS index found. Creating a new one...")

    documents = []  # ✅ re-initialize safely

    # 🔹 Ingest real documents
    if os.path.exists(config.data_path):
        for fname in os.listdir(config.data_path):
            if fname.lower().endswith(".txt"):
                full_path = os.path.join(config.data_path, fname)
                with open(full_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": fname,              # filename
                            "path": full_path,
                            "source_type": "static_txt"
                        }
                    )
                )

    # 🔹 Absolute fallback (never empty)
    if not documents:
        documents.append(
            Document(
                page_content="LMKR is a global technology and software company.",
                metadata={
                    "source": "fallback.txt",
                    "source_type": "static_txt"
                }
            )
        )

    # 🔹 Build + persist FAISS
    vectorstore = LC_FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(config.faiss_path)
    print("✅ FAISS index created and saved to disk.")






openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


def load_episodic_memory():
    if not os.path.exists(EPISODIC_MEMORY_PATH):
        return []
    try:
        with open(EPISODIC_MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_episode(thread_id: str, question: str, answer: str):
    mem = load_episodic_memory()
    mem.append({
        "thread_id": thread_id,
        "timestamp": datetime.utcnow().isoformat(),
        "q": question,
        "a": answer
    })
    mem = mem[-200:]
    with open(EPISODIC_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2)

def recall_episodes(thread_id: str, question: str, k: int = 2) -> str:
    mem = load_episodic_memory()

    q_tokens = [t for t in question.lower().split() if t]
    if not q_tokens:
        return ""

    hits = []
    for m in reversed(mem):
        if not isinstance(m, dict):
            continue

        m_tid = m.get("thread_id")
        m_q = str(m.get("q", "")).lower()
        m_a = str(m.get("a", ""))

        if not m_tid or m_tid != thread_id:
            continue

        if any(t in m_q for t in q_tokens):
            hits.append(m)
            if len(hits) >= k:
                break

    return "\n".join(
        f"Past Q: {h.get('q','')}\nPast A: {h.get('a','')}"
        for h in hits
    )


# =========================================================
# GUARDS
# =========================================================

warnings.filterwarnings("ignore", message="Could not obtain an event loop")
warnings.filterwarnings("ignore")

input_guard = Guard().use_many(
    DetectPII(pii_entities=config.PII_ENTITIES, on_fail=OnFailAction.FIX)
)

output_guard = Guard().use_many(
    ToxicLanguage(threshold=config.TOXIC_LANGUAGE_THRESHOLD, on_fail=OnFailAction.FIX),
)



def apply_input_guard(question: str) -> str:
    try:
        validation_result = input_guard.validate(question)
        return validation_result.validated_output
    except Exception:
        return question

def apply_output_guard(answer: str) -> str:
    try:
        validation_result = output_guard.validate(answer)
        return validation_result.validated_output
    except Exception as e:
        print(f"Output Blocked: {e}")
        return "I'm sorry, I cannot provide that information due to safety guidelines."


# =========================================================
# PYDANTIC MODELS (structured outputs)
# =========================================================

class QueryAugmentation(BaseModel):
    """Node: Retrieval Augmentation"""
    augmented_queries: List[str] = Field(
        description="List of 3 alternative versions of the user question to improve search coverage."
    )

class GeneratedAnswer(BaseModel):
    """Node: Generation"""
    answer: str = Field(description="The response to the user.")
    sources_used: List[str] = Field(default_factory=list, description="List of sources/links used.")

    @model_validator(mode="before")
    @classmethod
    def rescue_sources(cls, data):
        if isinstance(data, str):
            return {"answer": data, "sources_used": []}
        if isinstance(data, dict):
            answer_text = data.get("answer", "")
            if "Sources Used:" in answer_text and "sources_used" not in data:
                parts = answer_text.split("Sources Used:")
                data["answer"] = parts[0].strip()
                data["sources_used"] = ["Mentioned in answer"]
            return data
        return {"answer": str(data), "sources_used": []}

    @field_validator("answer", mode="before")
    @classmethod
    def flatten_list_answer(cls, v):
        if isinstance(v, list):
            return ", ".join(map(str, v))
        return v



class RouteDecision(BaseModel):
    """Router output model."""
    destination: Literal[
        "career_retrieve_node",
        "static_retrieve_node",
        "announcements_retrieve_node",
        "direct_node"
    ] = Field(
        description=(
            "Choose 'announcements_retrieve_node' for announcements, press releases, or latest updates about LMKR. "
            "Choose 'career_retrieve_node' for jobs/vacancies. "
            "Choose 'direct_node' for greetings/small talk unrelated to LMKR. "
            "Choose 'static_retrieve_node' for everything else."
        )
    )

class ProceduralRule(BaseModel):
    rule: str = Field(description="A concise instruction to improve future agent behavior.")

class AgentState(TypedDict):
    question: str
    context_chunks: List[dict]
    generated_answer: Optional[GeneratedAnswer]
    thread_id: str
    destination: str


# =========================================================
# LLM STRUCTURED HELPER (OpenAI parse)
# =========================================================

T = TypeVar("T", bound=BaseModel)

def query_llm_structured(prompt_text: str, response_model: Type[T]) -> Optional[T]:
    try:
        response = openai_client.beta.chat.completions.parse(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful corporate assistant for LMKR."},
                {"role": "user", "content": prompt_text}
            ],
            response_format=response_model,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print(f"OpenAI Structured Output Failed: {e}")
        return None


# =========================================================
# WEB SCRAPING (Careers + Announcements) with caching + links
# =========================================================

def _bs4_extract_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(config.tags_cleanup):
        tag.decompose()
    body = soup.find("body")
    text = body.get_text(separator="\n") if body else soup.get_text(separator="\n")
    return clean_text_content(text)

def fetch_and_clean_body(url: str, depth: int = 0) -> str:
    """
    Prefer Selenium Edge for JS-heavy pages. Fallback to requests if selenium isn't available.
    """
    if depth > 1:
        return ""

    try:
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options as EdgeOptions
        from selenium.webdriver.edge.service import Service as EdgeService

        print(f" Booting Headless Edge for: {url}")

        edge_options = EdgeOptions()
        edge_options.add_argument("--headless=new")
        edge_options.add_argument("--no-sandbox")
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--disable-dev-shm-usage")
        edge_options.add_argument("--log-level=3")
        edge_options.add_argument(f"--user-agent={config.web_user_agent}")
        edge_options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])

        edge_service = EdgeService()
        if os.name == "nt":
            try:
                edge_service.creation_flags = subprocess.CREATE_NO_WINDOW
            except Exception:
                pass

        driver = None
        try:
            driver = webdriver.Edge(options=edge_options, service=edge_service)
            driver.get(url)
            time.sleep(config.scrape_wait_time)
            return _bs4_extract_clean_text(driver.page_source)
        finally:
            if driver:
                driver.quit()

    except Exception as e:
        print(f"Selenium not available or failed ({e}). Falling back to requests...")

    # Requests fallback
    try:
        headers = {"User-Agent": config.web_user_agent}
        r = requests.get(url, headers=headers, timeout=config.scrape_timeout)
        r.raise_for_status()
        return _bs4_extract_clean_text(r.text)
    except Exception as e:
        print(f"Requests fallback failed: {e}")
        return ""
    

def extract_bamboohr_job_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/careers/" in href:
            if href.startswith("/"):
                href = "https://lmkr.bamboohr.com" + href
            links.append(href)

    return list(set(links))


@tool
def scrape_careers_tool() -> str:
    """
    Scrapes LMKR careers AND individual job descriptions.
    """
    print(f"Crawling BambooHR careers site...")

    listing_html = fetch_and_clean_body(config.career_url)
    job_links = extract_bamboohr_job_links(listing_html)

    print(f"Found {len(job_links)} job pages")

    all_jobs_text = []

    for url in job_links:
        print(f"   ↳ Scraping job: {url}")
        job_html = fetch_and_clean_body(url)
        if job_html:
            all_jobs_text.append(
                f"SOURCE: {url}\n\n{job_html}"
            )

    payload = (
        f"SOURCE: {config.career_url}\n"
        f"SCRAPED_AT: {datetime.now().isoformat()}\n\n"
        + "\n\n---\n\n".join(all_jobs_text)
    )

    save_to_file(payload, config.save_career_output)
    return payload


@tool
def scrape_announcements_fast_tool() -> str:
    """
    Scrapes LMKR announcements and press releases from the official website.
    """
    print(f"Tool Triggered: scraping announcements {config.announcements_url} ...")
    headers = {"User-Agent": config.web_user_agent}
    try:
        response = requests.get(config.announcements_url, headers=headers, timeout=config.scrape_timeout)
        response.raise_for_status()
        clean = _bs4_extract_clean_text(response.text)
        payload = f"SOURCE: {config.announcements_url}\nSCRAPED_AT: {datetime.now().isoformat()}\n\n{clean}"
        save_to_file(payload, config.save_announcement_output)
        return payload
    except Exception as e:
        print(f"Fast Scrape Error: {e}")
        return ""


# =========================================================
# GRAPH NODES (patched: renames + generator split)
# =========================================================

def input_guard_node(state: AgentState):
    print("\nNode: Input Guard (PII only)...")

    if not state.get("thread_id"):
        state["thread_id"] = f"thread_{int(time.time())}"

    question = apply_input_guard(state.get("question", ""))

    return {
        **state,
        "question": question,
        "thread_id": state["thread_id"],
    }



def router_node(state: AgentState):
    print("\nRouter: Prompt-based intent routing...")

    question = state["question"]

    namespace = ("instructions", "global")
    stored_rules = memory_store.search(namespace, limit=5)
    procedural_rules = "\n".join(
        f"- {m.value.get('rule','')}" for m in stored_rules
    ) if stored_rules else "None"

    episodic_history = recall_episodes(
        state["thread_id"],
        question,
        k=2
    )

    prompt = f"""
You are a routing controller for an LMKR assistant.

User Question:
{question}

Relevant Past Interactions:
{episodic_history or "None"}

Existing Procedural Routing Notes:
{procedural_rules}

Choose exactly ONE destination from:
- career_retrieve_node
- announcements_retrieve_node
- static_retrieve_node
- direct_node

Return JSON strictly matching the schema. No explanation.
"""

    decision = query_llm_structured(prompt, RouteDecision)

    destination = (
        decision.destination
        if decision and getattr(decision, "destination", None)
        else "static_retrieve_node"
    )

    print(f" Router (LLM) → {destination}")

    return {
        **state,
        "destination": destination
    }




def static_retrieve_node(state: AgentState):
    print("\nNode: Static Retrieve...")
    question = state["question"]

    prompt = f"User Question: {question}\nTask: Generate 3 different search query variations."
    structured_aug = query_llm_structured(prompt, QueryAugmentation)
    queries = [question] + (structured_aug.augmented_queries if structured_aug else [])

    all_docs = []

    for q in queries:
        docs = vectorstore.similarity_search(q, k=config.k_static)
        for d in docs:
            all_docs.append({
    "content": d.page_content,
    "metadata": {
        "source": d.metadata.get("source", "unknown.txt"),  # 👈 filename
        "path": d.metadata.get("path", ""),
        "type": "static_txt"
    }
})


    seen = set()
    unique_context = []

    for doc in all_docs:
        content = doc.get("content")
        if not content:
            continue
        if content in seen:
            continue
        seen.add(content)
        unique_context.append(doc)

    unique_context = unique_context[:config.k_static]

    save_to_file(
        "\n---\n".join(d["content"] for d in unique_context),
        config.static_output
    )

    return {"context_chunks": unique_context}



def career_retrieve_node(state: AgentState):
    print("\nNode: Career Retrieve...")
    question = state["question"]

    raw_text = (
        load_from_file(config.save_career_output)
        if is_file_fresh(config.save_career_output, config.scrape_after)
        else scrape_careers_tool.invoke({})
    )

    if not raw_text:
        return {"context_chunks": []}

    temp_vs = LC_FAISS.from_texts(
        split_text_into_chunks(raw_text),
        embeddings
    )

    retrieved_docs = temp_vs.similarity_search(
        question,
        k=config.k_careers
    )

    return {
    "context_chunks": [
        {
            "content": d.page_content,
            "metadata": {
                "source_url": config.career_url,
                "source_file": config.save_career_output,
                "source_type": "career_scrape"
            }
        }
        for d in retrieved_docs
    ]
}




def announcements_retrieve_node(state: AgentState):
    print("\nNode: Announcements Retrieve...")
    question = state["question"]

    raw_text = scrape_announcements_fast_tool.invoke({})
    if not raw_text:
        return {"context_chunks": []}

    temp_vs = LC_FAISS.from_texts(
        split_text_into_chunks(raw_text),
        embeddings
    )

    retrieved_docs = temp_vs.similarity_search(
        question,
        k=config.k_announcements
    )

    return {
        "context_chunks": [
            {
                "content": d.page_content,
                "metadata": {
    "source_url": config.announcements_url,
    "source_file": config.save_announcement_output,
    "source_type": "announcement_scrape"
}

            }
            for d in retrieved_docs
        ]
    }



def direct_node(state: AgentState):
    print("\nNode: Direct (LLM)...")

    question = state["question"]

    prompt = f"""
User Input: {question}
Instructions:
1. You are a helpful corporate assistant for LMKR.
2. Respond naturally to the greeting or conversational question.
3. Do NOT make up technical facts. Just be polite.
Return a short helpful reply.
"""

    response = query_llm_structured(prompt, GeneratedAnswer)

    if not response:
        response = GeneratedAnswer(answer="Hello! How can I help?", sources_used=["Direct"])
    else:
        response.sources_used = ["Direct"]

    # ✅ pass-through state
    return {
        **state,
        "generated_answer": response,
        "context_chunks": [],
        "destination": "direct_node",
    }




def _extract_links_from_context(chunks: List[str]) -> List[str]:
    # Pull "SOURCE: <url>" lines from context, plus any http(s) links.
    links = []
    for ch in chunks:
        for line in ch.splitlines():
            line = line.strip()
            if line.upper().startswith("SOURCE:"):
                url = line.split(":", 1)[1].strip()
                if url:
                    links.append(url)
        # also collect raw http links
        for token in ch.split():
            if token.startswith("http://") or token.startswith("https://"):
                links.append(token.strip(").,]}>\"'"))
    return list(dict.fromkeys(links))

def static_generate_node(state: AgentState):
    print("\nNode: Static Generate...")

    # ----- structured context -----
    context_data = "\n---\n".join(
        c["content"] for c in state.get("context_chunks", [])
    )

    # ----- episodic memory -----
    episodic_history = recall_episodes(
        state["thread_id"],
        state["question"],
        k=2
    )

    # ----- short-term conversation memory -----
    conversation = load_conversation_memory()

    today = datetime.now().strftime("%B %d, %Y")

    prompt = f"""
You are an expert assistant for LMKR.

Conversation History (recent):
{conversation}

Relevant Past Episodes:
{episodic_history}

STATIC Context Data:
{context_data}

Current Date: {today}
User Question: {state['question']}

Rules:
1. Answer using ONLY the Context Data.
2. Preserve sources if present.
3. If information is missing, say: "I do not have enough information."
4. Do NOT invent facts.

Return JSON strictly following the schema.
"""

    response = query_llm_structured(prompt, GeneratedAnswer)

    if response is None:
        response = GeneratedAnswer(
            answer="I do not have enough information.",
            sources_used=[]
        )

    # collect sources from metadata
    sources = []
    for c in state.get("context_chunks", []):
        if "source" in c.get("metadata", {}):
            sources.append(c["metadata"]["source"])

    response.sources_used = list(dict.fromkeys(sources))

    # ----- persist memories -----
    append_conversation(state["question"], response.answer)
    save_episode(state["thread_id"], state["question"], response.answer)

    return {"generated_answer": response}


def build_dynamic_sources(context_chunks):
    urls = []
    files = []

    for c in context_chunks:
        meta = c.get("metadata", {})
        if meta.get("source_url"):
            urls.append(meta["source_url"])
        if meta.get("source_file"):
            files.append(meta["source_file"])

    return {
        "urls": list(dict.fromkeys(urls)),
        "files": list(dict.fromkeys(files))
    }



def dynamic_generate_node(state: AgentState):
    print("\nNode: Dynamic Generate...")

    # ----- context -----
    context_data = "\n---\n".join(
        c["content"] for c in state.get("context_chunks", [])
        if isinstance(c, dict) and c.get("content")
    )

    # ----- memories -----
    episodic_history = recall_episodes(
        state["thread_id"],
        state["question"],
        k=2
    )

    conversation = load_conversation_memory()
    today = datetime.now().strftime("%B %d, %Y")

    # ----- build dynamic sources (URL + cache file) -----
    dynamic_sources = build_dynamic_sources(state.get("context_chunks", []))

    url_block = "\n".join(
        f"[{i+1}] {u}" for i, u in enumerate(dynamic_sources["urls"])
    )

    file_offset = len(dynamic_sources["urls"])

    file_block = "\n".join(
        f"[{file_offset+i+1}] {f}"
        for i, f in enumerate(dynamic_sources["files"])
    )

    sources_block = "\n".join(
        block for block in [url_block, file_block] if block
    )

    # ----- prompt -----
    prompt = f"""
You are an expert LMKR assistant.

Conversation History:
{conversation}

Relevant Past Episodes:
{episodic_history}

DYNAMIC Context Data (scraped):
{context_data}

Available Sources:
{sources_block}

Current Date: {today}
User Question: {state['question']}

Rules:
1. Use ONLY the provided context.
2. Cite factual statements using the source numbers like [1].
3. URLs and scrape cache files are valid sources.
4. If the answer is not present, say so explicitly.
5. Do NOT invent details.

Return JSON strictly following the schema.
"""

    response = query_llm_structured(prompt, GeneratedAnswer)

    if response is None:
        response = GeneratedAnswer(
            answer="I do not have enough information.",
            sources_used=[]
        )

    # ----- force sources (deterministic, no LLM guessing) -----
    response.sources_used = (
        dynamic_sources["urls"] +
        dynamic_sources["files"]
    )

    # ----- append Sources section (guaranteed) -----
    if response.sources_used:
        sources_text = "\n".join(
            f"[{i+1}] {s}"
            for i, s in enumerate(response.sources_used)
        )
        response.answer = f"{response.answer}\n\nSources:\n{sources_text}"

    # ----- persist memories -----
    append_conversation(state["question"], response.answer)
    save_episode(state["thread_id"], state["question"], response.answer)

    return {"generated_answer": response}



def output_guard_node(state: AgentState):
    print("\nNode: Output Guard (Safety Scan)...")
    generation = state["generated_answer"]
    generation.answer = apply_output_guard(generation.answer)
    return {"generated_answer": generation}



def save_memory_node(state: AgentState):
    if state.get("generated_answer"):
        save_episode(
            state["thread_id"],
            state["question"],
            state["generated_answer"].answer
        )
    return state






# =========================================================
# BUILD + COMPILE GRAPH (patched)
# =========================================================

workflow = StateGraph(AgentState)

workflow.add_node("input_guard_node", input_guard_node)
workflow.add_node("router_node", router_node)

# retrieval
workflow.add_node("static_retrieve_node", static_retrieve_node)
workflow.add_node("career_retrieve_node", career_retrieve_node)
workflow.add_node("announcements_retrieve_node", announcements_retrieve_node)
workflow.add_node("direct_node", direct_node)

# generators (split)
workflow.add_node("static_generate_node", static_generate_node)
workflow.add_node("dynamic_generate_node", dynamic_generate_node)

# post steps
workflow.add_node("output_guard_node", output_guard_node)
workflow.add_node("save_memory_node", save_memory_node)

workflow.set_entry_point("input_guard_node")

workflow.add_edge("input_guard_node", "router_node")

workflow.add_conditional_edges(
    "router_node",
    lambda x: x["destination"],
    {
        "career_retrieve_node": "career_retrieve_node",
        "announcements_retrieve_node": "announcements_retrieve_node",
        "static_retrieve_node": "static_retrieve_node",
        "direct_node": "direct_node",
    }
)

# retrieval -> generator
workflow.add_edge("static_retrieve_node", "static_generate_node")
workflow.add_edge("career_retrieve_node", "dynamic_generate_node")
workflow.add_edge("announcements_retrieve_node", "dynamic_generate_node")

# generators -> safety -> groundness
workflow.add_edge("static_generate_node", "output_guard_node")
workflow.add_edge("dynamic_generate_node", "output_guard_node")
workflow.add_edge("output_guard_node", "save_memory_node")



workflow.add_edge("direct_node", END)
workflow.add_edge("save_memory_node", END)


app = workflow.compile()
print("✅ Graph compiled successfully!")


# =========================================================
# CLI RUNNER
# =========================================================

def run_once(question: str, thread_id: str = "default_thread"):
    initial_state: AgentState = {
        "question": question,
        "context_chunks": [],
        "generated_answer": None,
        "thread_id": thread_id,
        "destination": ""
    }

    result = app.invoke(initial_state)

    return result

def run_cli(question: str, thread_id: str):
    result = run_once(question, thread_id)
    ga = result.get("generated_answer")

    if ga:
        print("\n========================")
        print("FINAL ANSWER:\n")
        print(ga.answer)

        if ga.sources_used:
            print("\nSOURCES / LINKS USED:")
            for s in ga.sources_used:
                print(f"- {s}")

        print("========================\n")



if __name__ == "__main__":
    tid = f"thread_{int(time.time())}"

    print("Type normally, or use :voice <seconds>  (example: :voice 6)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            break

        # -------- VOICE MODE --------
        if user_input.startswith(":voice"):
            try:
                seconds = float(user_input.split()[1])
            except Exception:
                seconds = 6.0

            spoken_text = listen(seconds)
            print(f"\n[ASR] You said: {spoken_text}\n")

            result = run_once(spoken_text, tid)
            ga = result.get("generated_answer")

            answer = ga.answer if ga else "I do not have enough information."

            print("FINAL ANSWER:\n")
            print(answer)
            print()

            # Speak WITHOUT sources
            speak_text = answer.split("\n\nSources:\n", 1)[0]
            speak(speak_text)

            continue

        # -------- TEXT MODE --------
        run_cli(user_input, tid)
