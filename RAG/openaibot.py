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
from voice_io import transcribe_audio as listen
from voice_io import speak_to_file as speak
from concurrent.futures import ThreadPoolExecutor

MEMORY_EXECUTOR = ThreadPoolExecutor(max_workers=4)
from openai import APIConnectionError, RateLimitError, APITimeoutError



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


EMBED_CACHE = {}
def embed_once(text: str):
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]

    try:
        vec = embeddings.embed_query(text)
        EMBED_CACHE[text] = vec
        return vec

    except APIConnectionError as e:
        print(f"⚠️ Embedding failed, skipping query: {e}")
        return None

SCRAPE_FRESH_HOURS = 3

# =========================================================
# CONFIG (inline)
# =========================================================

class config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Embeddings
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    # Vector DB (ABSOLUTE PATHS – REQUIRED)
    faiss_path = os.getenv(
        "faiss_path",
        r"C:\Users\afarooq\Downloads\lmkr-company-website (2)\lmkr-company-website\RAG\faiss_data"
    )

    data_path = os.getenv(
        "data_path",
        r"C:\Users\afarooq\Downloads\lmkr-company-website (2)\lmkr-company-website\RAG\lmkr_data"
    )

    static_output = os.getenv("static_output", "context_debug.txt")

    # Retrieval
    static_k = int(os.getenv("static_k", "5"))
    careers_k = int(os.getenv("careers_k", "20"))
    announcements_k = int(os.getenv("announcements_k", "5"))

    # Scraping URLs (set yours if different)
    careers_url = os.getenv("careers_url", "https://lmkr.bamboohr.com/careers")
    announcements_url = os.getenv("announcements_url", "https://lmkr.com/announcements/")

    career_faiss_path = "career_faiss"
    announcement_faiss_path = "announcement_faiss"
    



    # Cache files
    careers_output_file = os.getenv("careers_output_file", "careers_cache.txt")
    announcement_output_file = os.getenv("announcement_output_file", "announcements_cache.txt")

    scrape_after = int(os.getenv("scrape_after", "3"))
    scrape_timeout = int(os.getenv("scrape_timeout", "20"))

    # Selenium
    wait_time = float(os.getenv("wait_time", "2"))
    web_user_agent = os.getenv(
        "web_user_agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )

    # HTML cleanup
    tags_cleanup = ["script", "style", "noscript", "svg"]

    # Guardrails
    PII_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER"]
    TOXIC_LANGUAGE_THRESHOLD = float(os.getenv("TOXIC_LANGUAGE_THRESHOLD", "0.8"))



def load_or_build_faiss(path, texts, embeddings):
    if os.path.exists(path):
        try:
            return LC_FAISS.load_local(
                path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            pass

    vs = LC_FAISS.from_texts(texts, embeddings)
    vs.save_local(path)
    return vs

def trim_context_preserve_accuracy(chunks, max_chars=4000):
    trimmed = []
    total = 0

    for c in chunks:
        content = c["content"]
        if total + len(content) > max_chars:
            break
        trimmed.append(c)
        total += len(content)

    return trimmed

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
        print("FAISS index loaded from disk.")
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")

# 2️⃣ If not found or failed → build & persist
if vectorstore is None:
    print("No FAISS index found. Creating a new one...")

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
    print("FAISS index created and saved to disk.")






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
            "Choose 'career_retrieve_node' for jobs/vacancies,hiring,careers. "
            "Choose direct_node for greetings, small talk, or general world-knowledge questions that are not specific to LMKR. "
            "Choose static_retrieve_node ONLY if the question is about LMKR, its products, platforms, services, or company information.\n"
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

def Structured_output(prompt_text: str, response_model: Type[T]) -> Optional[T]:
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

    except (APIConnectionError, APITimeoutError, RateLimitError) as e:
        print(f"⚠️ OpenAI temporary failure: {e}")
        return None

    except Exception as e:
        print(f"❌ OpenAI Structured Output Failed: {e}")
        return None


# =========================================================
# WEB SCRAPING (Careers + Announcements) with caching + links
# =========================================================

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

    # Try Selenium (Edge). If anything fails, fallback to requests.
    try:
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options as EdgeOptions
        from selenium.webdriver.edge.service import Service as EdgeService

        print(f" 🖥️ Booting Headless Edge for: {url}")

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
            # Hide windows driver logs/popups where possible
            try:
                edge_service.creation_flags = subprocess.CREATE_NO_WINDOW
            except Exception:
                pass

        driver = None
        try:
            driver = webdriver.Edge(options=edge_options, service=edge_service)
            driver.get(url)
            time.sleep(config.wait_time)
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



@tool
def scrape_careers_tool() -> str:
    """
    Scrapes LMKR careers AND individual job descriptions.
    """
    print("Crawling BambooHR careers site...")

    listing_html = fetch_and_clean_body(config.careers_url)
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
        f"SOURCE: {config.careers_url}\n"
        f"SCRAPED_AT: {datetime.now().isoformat()}\n\n"
        + "\n\n---\n\n".join(all_jobs_text)
    )

    save_to_file(payload, config.careers_output_file)
    return payload



@tool
def scrape_announcements_tool() -> str:
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
        save_to_file(payload, config.announcement_output_file)
        return payload
    except Exception as e:
        print(f"Fast Scrape Error: {e}")
        return ""


# =========================================================
# GRAPH NODES (patched: renames + generator split)
# =========================================================

def input_guard_node(state: AgentState):
    print("\nNode: Input Guard...")

    if not state.get("thread_id"):
        state["thread_id"] = f"thread_{int(time.time())}"

    question = apply_input_guard(state.get("question", ""))

    return {
        **state,
        "question": question,
        "thread_id": state["thread_id"],
    }



def router_node(state: AgentState):
    print("\nRouter: LLM intent routing...")

    question = state["question"]

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

Choose exactly ONE destination from:
- career_retrieve_node
- announcements_retrieve_node
- static_retrieve_node
- direct_node

Return JSON strictly matching the schema. No explanation.
"""

    decision = Structured_output(prompt, RouteDecision)

    destination = (
        decision.destination
        if decision and getattr(decision, "destination", None)
        else "static_retrieve_node"
    )

    print(f" Router → {destination}")

    return {**state, "destination": destination}





def static_retrieve_node(state: AgentState):
    print("\nNode: Static Retrieve...")
    question = state["question"]

    # --------------------------------------------------
    # SAFE QUERY AUGMENTATION (NO UNBOUND VARIABLES)
    # --------------------------------------------------
    queries = [question]  # ALWAYS defined

    if len(question.split()) >= 6:
        aug_prompt = (
            f"User Question: {question}\n"
            f"Task: Generate 3 different search query variations."
        )

        structured_aug = Structured_output(aug_prompt, QueryAugmentation)

        if structured_aug and structured_aug.augmented_queries:
            queries.extend(structured_aug.augmented_queries)

    # --------------------------------------------------
    # VECTOR SEARCH (EMBED ONCE PER QUERY VARIANT)
    # --------------------------------------------------
    all_docs = []

    for q in queries:
        query_embedding = embed_once(q)
        if query_embedding is None:
            continue

        docs = vectorstore.similarity_search_by_vector(
            query_embedding,
            k=config.static_k
        )

        

        for d in docs:
            all_docs.append({
                "content": d.page_content,
                "metadata": {
                    "source": d.metadata.get("source", "unknown.txt"),
                    "path": d.metadata.get("path", ""),
                    "type": "static_txt"
                }
            })

    # --------------------------------------------------
    # DEDUP + TOP-K
    # --------------------------------------------------
    seen = set()
    unique_context = []

    for doc in all_docs:
        content = doc.get("content")
        if not content or content in seen:
            continue
        seen.add(content)
        unique_context.append(doc)

    unique_context = unique_context[:config.static_k]

    # --------------------------------------------------
    # DEBUG SAVE (OPTIONAL)
    # --------------------------------------------------
    save_to_file(
        "\n---\n".join(d["content"] for d in unique_context),
        config.static_output
    )

    return {"context_chunks": unique_context}




def career_retrieve_node(state: AgentState):
    print("\nNode: Career Retrieve...")
    question = state["question"]

    if is_file_fresh(config.careers_output_file, SCRAPE_FRESH_HOURS):

        raw_text = load_from_file(config.careers_output_file)
    else:
        raw_text = scrape_careers_tool.invoke({})


    if not raw_text:
        return {"context_chunks": []}

    chunks = split_text_into_chunks(raw_text)

    career_vs = load_or_build_faiss(
    config.career_faiss_path,
    chunks,
    embeddings
)

    retrieved_docs = career_vs.similarity_search(
        question,
        k=config.careers_k
    )

    return {
        "context_chunks": [
            {
                "content": d.page_content,
                "metadata": {
                    "source_url": config.careers_url,
                    "source_file": config.careers_output_file,
                    "source_type": "career_scrape"
                }
            }
            for d in retrieved_docs
        ]
    }








def announcements_retrieve_node(state: AgentState):
    print("\nNode: Announcements Retrieve...")
    question = state["question"]

    if is_file_fresh(config.announcement_output_file, SCRAPE_FRESH_HOURS):
        raw_text = load_from_file(config.announcement_output_file)
    else:
        raw_text = scrape_announcements_tool.invoke({})


    chunks = split_text_into_chunks(raw_text)

    ann_vs = load_or_build_faiss(
    config.announcement_faiss_path,
    chunks,
    embeddings
)

    retrieved_docs = ann_vs.similarity_search(
        question,
        k=config.announcements_k
    )

    return {
        "context_chunks": [
            {
                "content": d.page_content,
                "metadata": {
    "source_url": config.announcements_url,
    "source_file": config.announcement_output_file,
    "source_type": "announcement_scrape"
}

            }
            for d in retrieved_docs
        ]
    }


def is_small_talk(question: str) -> bool:
    q = question.lower().strip()

    small_talk_phrases = [
        "hi", "hello", "hey",
        "how are you",
        "how's it going",
        "good morning",
        "good afternoon",
        "good evening",
        "thank you",
        "thanks",
        "bye",
        "goodbye"
    ]

    # exact or starts-with match (prevents false positives)
    return any(
        q == p or q.startswith(p + " ")
        for p in small_talk_phrases
    )


def direct_node(state: AgentState):
    print("\nNode: Direct...")

    question = state["question"]

    # ✅ Allow greetings / small talk
    if is_small_talk(question):
        response = Structured_output(
            f"""
User Input: {question}
You are a polite corporate assistant for LMKR.
Respond briefly and naturally.
Return plain text only.
""",
            GeneratedAnswer
        )

        if not response:
            response = GeneratedAnswer(answer="Hello! How can I help you with LMKR today?", sources_used=[])

        return {
            **state,
            "generated_answer": response,
            "context_chunks": [],
            "destination": "direct_node",
        }

    # ❌ Block general knowledge / definitions
    return {
        **state,
        "generated_answer": GeneratedAnswer(
            answer="Sorry, I can only help with questions about LMKR, its products, and its services.",
            sources_used=[]
        ),
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
    safe_chunks = trim_context_preserve_accuracy(
    state.get("context_chunks", []),
    max_chars=4000
)

    context_data = "\n---\n".join(c["content"] for c in safe_chunks)


    episodic_future = MEMORY_EXECUTOR.submit(
    recall_episodes,
    state["thread_id"],
    state["question"],
    2
)

    conversation_future = MEMORY_EXECUTOR.submit(load_conversation_memory)

    episodic_history = episodic_future.result()
    conversation = conversation_future.result()



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
1. You MUST answer ONLY using the provided Context Data.
2. You are FORBIDDEN from using general knowledge.
3. If the answer is not explicitly stated in the Context Data,
   respond exactly with:
   "Sorry, I do not have enough information. Feel free to ask me more about LMKR!"

4. If the Context Data partially answers the question, provide a concise, factual answer based on it.
5. Only say "Sorry, I do not have enough information. Feel free to ask me more about LMKR!" if the Context Data is empty or completely irrelevant.
6. Do NOT invent facts, numbers, dates, or claims not supported by the context.
7. Return your answer as PLAIN TEXT ONLY. Do NOT use markdown formatting.


Return JSON strictly following the schema.
"""

    response = Structured_output(prompt, GeneratedAnswer)

    if response is None:
        response = GeneratedAnswer(
            answer="Sorry, I do not have enough information. Feel free to ask me more about LMKR!",
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

    # --------------------------------------------------
    # CONTEXT (trimmed for safety + speed)
    # --------------------------------------------------
    safe_chunks = trim_context_preserve_accuracy(
        state.get("context_chunks", []),
        max_chars=4000
    )

    context_data = "\n---\n".join(
        c["content"] for c in safe_chunks if c.get("content")
    )

    # --------------------------------------------------
    # PARALLEL MEMORY LOAD
    # --------------------------------------------------
    episodic_future = MEMORY_EXECUTOR.submit(
        recall_episodes,
        state["thread_id"],
        state["question"],
        2
    )
    conversation_future = MEMORY_EXECUTOR.submit(load_conversation_memory)

    episodic_history = episodic_future.result()
    conversation = conversation_future.result()

    today = datetime.now().strftime("%B %d, %Y")

    # --------------------------------------------------
    # SOURCE BLOCK (urls + cache files)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # DETECT "LIST ALL JOBS" CAREERS QUERY
    # --------------------------------------------------
    q_lower = state["question"].lower()
    list_jobs_mode = any(phrase in q_lower for phrase in [
        "what jobs are available",
        "list jobs",
        "open positions",
        "current openings",
        "available positions",
        "job openings"
    ])

    # --------------------------------------------------
    # PROMPT (CONDITIONAL)
    # --------------------------------------------------
    if list_jobs_mode:
        prompt = f"""
You are an LMKR careers assistant.

DYNAMIC Context Data (scraped careers content):
{context_data}

Instructions:
1. Identify ALL distinct job openings mentioned in the context.
2. List EVERY available job — do not summarize or filter.
3. Use ONLY the provided context. Do NOT infer or invent jobs.
4. If available, include location and department.
5. Output must be plain text, one job per line, like:
   Job Title – Location (Department)
6. If location or department is missing, omit it.
7. If no jobs are found, clearly say so.

Return JSON strictly following the schema.
"""
    else:
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
6. Return your answer as PLAIN TEXT ONLY. Do NOT use markdown formatting.

Return JSON strictly following the schema.
"""

    # --------------------------------------------------
    # GENERATION
    # --------------------------------------------------
    response = Structured_output(prompt, GeneratedAnswer)

    if response is None:
        response = GeneratedAnswer(
            answer="Sorry, I do not have enough information. Feel free to ask me more about LMKR!",
            sources_used=[]
        )

    # --------------------------------------------------
    # FORCE SOURCES (DETERMINISTIC)
    # --------------------------------------------------
    response.sources_used = response.sources_used or (
        dynamic_sources["urls"] + dynamic_sources["files"]
    )

    # --------------------------------------------------
    # PERSIST MEMORY
    # --------------------------------------------------
    append_conversation(state["question"], response.answer)

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
    # -------------------------------
    # 🚫 Guard against empty/weak input
    # -------------------------------
    if not question or len(question.strip()) < 3:
        return {
            "generated_answer": GeneratedAnswer(
                answer="Please ask a more specific question.",
                sources_used=[]
            )
        }

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