## LMKR Intelligent Assistant (RAG + LangGraph)

An end-to-end corporate AI assistant for LMKR that combines Retrieval-Augmented Generation (RAG), web scraping, episodic memory, and LLM routing using LangGraph.

The assistant can:

* Answer questions from static company documents

* List and filter career openings (BambooHR)

* Answer questions about announcements / press releases

* Handle greetings and small talk

* Maintain short-term and episodic memory

* Enforce PII and toxicity guardrails

* Run in text or voice mode

## Features
### Intelligent Routing

Uses an LLM-based router to decide whether a query should go to:

* Static knowledge base

* Careers (jobs)

* Announcements

* Direct conversational response

### Retrieval-Augmented Generation (RAG)

* FAISS vector store (persistent on disk)

* Chunked static documents

* Query augmentation for better recall

* Dimension-safe FAISS loading

### Memory

* Conversation memory (last 50 turns)

* Episodic memory (thread-scoped recall)

* Memory used during generation for continuity

### Live Web Scraping

* Careers: BambooHR via Selenium (JS-safe)

* Announcements: lmkr.com via requests

* Smart caching with freshness checks

### Guardrails

* Input: automatic PII detection & fixing

* Output: toxic language detection & blocking

### Voice Support

* Speech-to-text input

* Text-to-speech output

## Built using:

* LangGraph (stateful agent orchestration)

* LangChain FAISS

* OpenAI structured outputs

* Pydantic models for safety & parsing

## Requirements

Python 3.11-3.12 recommended

Environment Variables

Create a .env file:
`OPENAI_API_KEY=your_openai_key_here`

`LLM_MODEL=gpt-4o-mini`

`LLM_MAX_TOKENS=800`

`LLM_TEMPERATURE=0.2`

`EMBEDDINGS_MODEL=text-embedding-3-small`

`EMBEDDING_DIMENSION=1536`

`faiss_path=path/to/faiss_data
data_path=path/to/static_txt_files`

## Dependencies 

Install dependencies using: `pip install -r requirements.txt`

## Run CLI

`python main.py`

OR

## Run Flask Endpoints and test through Postman

`python flaskapi.py`

## Example Queries

Tested and working Queries:

'What is LMKT?'

'What job is available in Lahore?'

'What vacancies are open for Business?'

'What vacancies are open for finance?'

'What is GVERSE and TRVERSE?'

'Who is the CEO and when was the company founded?'

'Does lmkr operate in Malaysia and Japan?'

'How many employees are working at lmkr?'

'Where is it located in islamabad?'

'I want to apply at lmkr'

'What is latest news at lmkr?'

'My number is 3433430922, can you contact me?' (this query wouldnt work because of the DetectPII guardrail)

## Memory Behaviour 

* Conversation history influences the response
  
* Episodic memory is scoped per thread_id

* Automatically saved after each response

* Used during routing and generation

