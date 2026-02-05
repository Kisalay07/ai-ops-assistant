# AI Ops Assistant 🤖⚡

AI Ops Assistant is a local backend service that uses a **multi-agent LLM system** to produce end-to-end answers by combining reasoning with real-world data from third-party APIs.

The project runs **locally on localhost** and demonstrates agent planning, tool execution, and verification with **no hard-coded responses**.

---

## 🏗 Architecture Overview (Agents + Tools)

The system uses a **multi-agent pipeline**:

### Agents

1. **Planner Agent**
   - Interprets the user prompt
   - Breaks the request into steps
   - Decides which tools are required

2. **Executor Agent**
   - Executes the plan
   - Calls external tools (news, weather, etc.)
   - Collects real-world data

3. **Verifier Agent**
   - Checks that the response:
     - satisfies the original request
     - includes required data
     - is complete and coherent

### Tools

- Tools are the **only components** allowed to access external APIs
- Each tool has a single responsibility (news, weather, etc.)
- Agents never call external APIs directly

This separation ensures clean orchestration, extensibility, and correctness.

---

## 🔌 Integrated APIs

The system integrates **real third-party services**:

- **Groq**
  - LLM inference
  - Structured outputs / tool selection
  - Default model:
    ```
    meta-llama/llama-4-scout-17b-16e-instruct
    ```

- **NewsAPI.org**
  - Live global and national news headlines

- **OpenWeatherMap**
  - Current weather conditions and forecasts

---

> The project uses a **flat structure** and must be run from the folder that contains `main.py`.

---

## 🧑‍💻 Setup Instructions (Run Locally on localhost)

Follow these steps **top to bottom**.

---

### 1️⃣ Prerequisites

- Python **3.9+**
- pip
- API keys for:
  - Groq
  - NewsAPI.org
  - OpenWeatherMap

Verify Python installation:

```bash
python --version

2️⃣ Create and activate a virtual environment

From the project root:

python -m venv .venv


Activate it:

Windows

.venv\Scripts\activate


macOS / Linux

source .venv/bin/activate

3️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

🔐 Environment Variables

Create a .env file in the project root.

.env.example
# Groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# NewsAPI.org
NEWSAPI_KEY=your_newsapi_key_here

# OpenWeatherMap
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Server
HOST=127.0.0.1
PORT=8000

# Runtime
LOG_LEVEL=INFO


▶️ Running the Project (One Command)

Run this command from the folder that contains main.py:

uvicorn main:app --reload


The service will be available at:

http://127.0.0.1:8000

## 🧪 Example Prompts

Use these prompts to test the full **multi-agent + tool** pipeline:

**Daily briefing**  
> “Give me a daily briefing with top national and global news and today’s weather in New Delhi.”

**News analysis**  
> “Summarize today’s most important India-related technology or startup news and explain why it matters.”

**Weather lookup**  
> “What’s the current weather in Mumbai and should I expect heavy rain today?”

---

## ⚖ Known Limitations / Tradeoffs

- **Stateless execution**  
  No long-term memory between requests

- **External API dependency**  
  Latency and availability depend on third-party services

- **No caching**  
  Repeated requests may re-fetch the same data

- **Single-process local server**  
  No built-in scaling, authentication, or rate limiting

- **Local-first**  
  Designed for development, not production deployment
