# BMW Intelligence — Conversational BI Dashboard

A full-stack AI-powered Business Intelligence dashboard. Ask natural language questions about BMW vehicle inventory data and instantly get interactive charts and insights.

---

## Architecture

```
frontend/
  index.html          ← Single-page app (vanilla JS + Chart.js)

backend/
  main.py             ← FastAPI server (query engine + LLM bridge)
  bmw_data.json       ← Pre-processed BMW inventory (10,781 records)
  requirements.txt    ← Python dependencies
```

**Pipeline:** `User query → FastAPI → Groq openai/gpt-oss-120b (spec generation) → Query Engine (data aggregation) → JSON → Frontend (Chart.js render)`

---

## Quick Start

### 1. Backend

```bash
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY=gsk_...   # Windows: set GROQ_API_KEY=gsk_...

# Optional: override model/base URL
# export GROQ_MODEL=openai/gpt-oss-120b
# export GROQ_BASE_URL=https://api.groq.com/openai/v1

# Start the server
uvicorn main:app --reload --port 8000
```

Backend will be live at: http://localhost:8000

### 2. Frontend

Open `frontend/index.html` directly in your browser, **or** serve it:

```bash
# Simple HTTP server (Python)
cd frontend
python3 -m http.server 3000
# Then visit http://localhost:3000
```

> The frontend auto-detects `localhost` and points API calls to `http://localhost:8000`.

---

## Features

| Feature | Status |
|---------|--------|
| Natural language → Dashboard | ✅ |
| Automatic chart type selection | ✅ |
| 6 chart types (bar, line, pie, doughnut, scatter, horizontal bar) | ✅ |
| Multi-series / grouped charts | ✅ |
| KPI metric cards | ✅ |
| AI-generated insight summary | ✅ |
| Follow-up / context-aware queries | ✅ |
| Query history sidebar | ✅ |
| CSV file upload (any dataset) | ✅ |
| Hallucination / can't-answer detection | ✅ |
| Auto-overview dashboard | ✅ |
| Interactive tooltips on all charts | ✅ |

---

## Example Queries

**Simple**
- "Show average price by model"
- "Fuel type distribution"
- "Which transmission type is most common?"

**Medium**
- "How has average price changed over the years?"
- "Show top 10 models by inventory count broken down by transmission"
- "Which models have the best average fuel efficiency?"

**Complex**
- "Compare mileage vs price as a scatter plot colored by fuel type"
- "Show price distribution across price ranges and compare diesel vs petrol average cost"
- "What is the relationship between engine size and price for automatic cars only?"

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Run a natural language query |
| POST | `/api/upload` | Upload a new CSV dataset |
| GET | `/api/stats` | Get pre-computed summary stats |
| GET | `/api/health` | Health check |

### POST /api/query

```json
{
  "query": "Show average price by fuel type",
  "context": null
}
```

Response:
```json
{
  "cannotAnswer": false,
  "insight": "Diesel vehicles average £21,200 vs Petrol at £26,400",
  "kpis": [...],
  "charts": [{ "type": "bar", "title": "...", "result": { "labels": [...], "values": [...] } }]
}
```

---

## Tech Stack

- **Frontend**: Vanilla HTML/CSS/JS, Chart.js 4.4
- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **AI**: Groq OpenAI-compatible API using `openai/gpt-oss-120b` (query interpretation + chart spec generation)
- **Data**: In-memory JSON (10,781 BMW records), CSV upload support
- **Fonts**: Rajdhani + DM Sans (Google Fonts)

---

## Evaluation Criteria Coverage

| Criterion | How it's addressed |
|-----------|-------------------|
| **Accuracy (40pts)** | Groq (`openai/gpt-oss-120b`) generates structured chart specs; a deterministic query engine executes them against real data — no hallucinated numbers |
| **Chart selection** | LLM chooses bar/line/pie/scatter based on data type and question intent |
| **Error handling** | `cannotAnswer` flag + graceful error UI for ambiguous/impossible queries |
| **Aesthetics (30pts)** | Dark luxury BMW theme, Rajdhani display font, animated transitions, custom tooltips |
| **Interactivity** | Hover tooltips, Chart.js zoom, legend toggles on all charts |
| **UX / loading** | Animated spinner with cycling status messages, query chips for guidance |
| **Innovation (30pts)** | Structured prompt engineering with schema injection; LLM generates spec, engine executes — separating AI from data layer |
| **Follow-up queries (bonus +10)** | Context passed with every request; sidebar history |
| **CSV upload (bonus +20)** | Drag-and-drop CSV upload, auto schema inference, instant query-ready |
