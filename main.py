"""
BMW BI Dashboard — FastAPI Backend
Runs on: uvicorn main:app --reload --port 8000
"""

import json
import os
import re
import csv
import io
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

# ─── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="BMW BI Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "bmw_data.json"

# ─── In-memory data store ─────────────────────────────────────────────────────
class DataStore:
    def __init__(self):
        self.data: list[dict] = []
        self.schema: dict = {}
        self.source_name: str = "BMW Vehicle Inventory"
        self._load_default()

    def _load_default(self):
        if not DATA_FILE.exists():
            self.data = []
            self.schema = {}
            self.source_name = "No dataset loaded"
            return
        with open(DATA_FILE) as f:
            self.data = json.load(f)
        self._infer_schema()

    def load_csv(self, content: str, filename: str):
        # Normalize line endings and parse with newline="" to avoid csv parser edge cases.
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        stream = io.StringIO(normalized, newline="")

        sample = normalized[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(stream, dialect=dialect, restkey="__overflow__", skipinitialspace=True)
        rows = []

        def _clean_cell(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                # DictReader returns list values for overflow columns.
                value = " ".join(str(x) for x in value if x is not None)
            return str(value).strip()

        def _to_number(value: str):
            # Accept common numeric text formats like "£12,340".
            s = str(value).strip()
            if not s:
                return None
            s = re.sub(r"[^0-9.\-]", "", s)
            if s in ("", "-", ".", "-."):
                return None
            try:
                n = float(s)
                return int(n) if n.is_integer() else n
            except ValueError:
                return None

        try:
            for row in reader:
                cleaned = {}
                for k, v in row.items():
                    key = (k or "extra_column").strip() or "extra_column"
                    if key == "__overflow__" and isinstance(v, list):
                        # Keep overflow tokens structured so we can recover columns later.
                        cleaned[key] = [str(x).strip() for x in v if str(x).strip()]
                        continue
                    v = _clean_cell(v)
                    if v == "":
                        cleaned[key] = ""
                        continue
                    try:
                        if "." in v:
                            cleaned[key] = float(v)
                        else:
                            cleaned[key] = int(v)
                    except ValueError:
                        cleaned[key] = v
                rows.append(cleaned)
        except csv.Error as e:
            raise ValueError(f"CSV format issue: {e}") from e

        if not rows:
            raise ValueError("CSV appears empty or has no readable rows.")

        # Expand DictReader overflow into stable columns (extra_column_1..N).
        overflow_present = any("__overflow__" in r for r in rows)
        if overflow_present:
            max_extra = 0
            for r in rows:
                vals = r.get("__overflow__", [])
                if isinstance(vals, list):
                    max_extra = max(max_extra, len(vals))

            for r in rows:
                vals = r.pop("__overflow__", [])
                if not isinstance(vals, list):
                    vals = [str(vals)] if vals else []
                for i in range(max_extra):
                    col = f"extra_column_{i+1}"
                    cell = vals[i].strip() if i < len(vals) else ""
                    num = _to_number(cell)
                    r[col] = num if num is not None else cell

        # If no explicit numeric price column exists, try to infer from extra columns.
        has_price = any("price" in r for r in rows)
        if not has_price:
            extra_cols = sorted({k for r in rows for k in r.keys() if k.startswith("extra_column_")})
            best_col = None
            best_score = 0.0
            for col in extra_cols:
                numeric_vals = []
                for r in rows:
                    n = _to_number(r.get(col, ""))
                    if n is not None:
                        numeric_vals.append(float(n))
                if not rows:
                    continue
                ratio = len(numeric_vals) / len(rows)
                if not numeric_vals:
                    continue
                avg = sum(numeric_vals) / len(numeric_vals)
                # Prefer mostly numeric columns with car-price-like scale.
                if ratio >= 0.6 and 100 <= avg <= 1_000_000 and ratio > best_score:
                    best_score = ratio
                    best_col = col

            if best_col:
                for r in rows:
                    n = _to_number(r.get(best_col, ""))
                    if n is not None:
                        r["price"] = n

        self.data = rows
        self.source_name = filename
        self._infer_schema()
        return len(rows)

    def _infer_schema(self):
        if not self.data:
            return
        sample = self.data[0]
        schema = {}
        for col, val in sample.items():
            if isinstance(val, float):
                schema[col] = "float"
            elif isinstance(val, int):
                schema[col] = "integer"
            else:
                unique_vals = list(set(str(r.get(col, "")) for r in self.data[:200]))
                schema[col] = {
                    "type": "categorical",
                    "sample_values": unique_vals[:8],
                    "cardinality": len(set(str(r.get(col, "")) for r in self.data))
                }
        self.schema = schema

    def schema_summary(self) -> str:
        lines = [f"Dataset: {self.source_name} ({len(self.data):,} records)"]
        lines.append("Columns:")
        for col, info in self.schema.items():
            if isinstance(info, dict):
                vals = ", ".join(info["sample_values"][:5])
                lines.append(f"  - {col} [categorical, {info['cardinality']} unique values]: e.g. {vals}")
            else:
                vals = [r.get(col) for r in self.data[:5] if r.get(col) is not None]
                lines.append(f"  - {col} [{info}]: e.g. {vals}")
        return "\n".join(lines)


store = DataStore()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def get_llm_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing GROQ_API_KEY environment variable.",
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


# ─── Query execution engine ────────────────────────────────────────────────────
def execute_spec(spec: dict, data: list[dict]) -> dict:
    """Execute a chart spec against the in-memory data and return chart-ready data."""
    chart_type = spec.get("type", "bar")
    group_by = spec.get("groupBy")
    second_group = spec.get("secondGroupBy")
    metric = spec.get("metric", "count")
    filters = spec.get("filters", [])
    top_n = spec.get("topN")
    sort_desc = spec.get("sortDesc", True)

    # Apply filters
    rows = data
    for f in filters:
        field = f.get("field")
        op = f.get("op", "eq")
        val = f.get("value")
        if not field:
            continue
        if op == "eq":
            rows = [r for r in rows if str(r.get(field, "")).strip() == str(val)]
        elif op == "gte":
            rows = [r for r in rows if _num(r.get(field)) >= _num(val)]
        elif op == "lte":
            rows = [r for r in rows if _num(r.get(field)) <= _num(val)]
        elif op == "in":
            rows = [r for r in rows if str(r.get(field, "")).strip() in [str(v) for v in val]]

    # Scatter plot — no aggregation
    if chart_type == "scatter":
        x_field = spec.get("xField", group_by)
        y_field = spec.get("yField", metric)
        color_field = spec.get("colorField")
        import random; random.seed(42)
        sample = random.sample(rows, min(600, len(rows)))
        points = []
        for r in sample:
            pt = {"x": _num(r.get(x_field, 0)), "y": _num(r.get(y_field, 0))}
            if color_field:
                pt["group"] = str(r.get(color_field, ""))
            points.append(pt)
        return {"type": "scatter", "data": points, "xLabel": x_field, "yLabel": y_field}

    # Aggregation
    if not group_by:
        return {"type": chart_type, "data": [], "labels": [], "values": []}

    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        key = str(r.get(group_by, "Unknown")).strip()
        groups[key].append(r)

    if second_group:
        all_subs = sorted(set(str(r.get(second_group, "")).strip() for r in rows))[:7]
        entries = []
        for key, group_rows in groups.items():
            entry = {"label": key, "count": len(group_rows)}
            for sub in all_subs:
                sub_rows = [r for r in group_rows if str(r.get(second_group, "")).strip() == sub]
                entry[sub] = _aggregate(sub_rows, metric)
            entries.append(entry)
        entries = _sort_entries(entries, all_subs[0] if all_subs else "count", sort_desc)
        if top_n:
            entries = entries[:top_n]
        return {"type": chart_type, "data": entries, "subKeys": all_subs, "groupBy": group_by}

    entries = []
    for key, group_rows in groups.items():
        entries.append({"label": key, "value": _aggregate(group_rows, metric), "count": len(group_rows)})

    entries = _sort_entries(entries, "value", sort_desc)
    if group_by in ("year",):
        entries.sort(key=lambda e: e["label"])
    if top_n:
        entries = entries[:top_n]

    labels = [e["label"] for e in entries]
    values = [e["value"] for e in entries]
    return {"type": chart_type, "labels": labels, "values": values, "entries": entries}


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _aggregate(rows: list, metric: str) -> float:
    if not rows:
        return 0
    if metric == "count":
        return len(rows)
    # Extract field name from metric like "avg:price", "sum:price", "max:price"
    if ":" in metric:
        agg, field = metric.split(":", 1)
    else:
        agg, field = "avg", metric

    vals = [_num(r.get(field, 0)) for r in rows]
    if agg == "avg":
        return round(sum(vals) / len(vals), 1)
    elif agg == "sum":
        return sum(vals)
    elif agg == "max":
        return max(vals)
    elif agg == "min":
        return min(vals)
    return len(rows)


def _sort_entries(entries, key, desc=True):
    return sorted(entries, key=lambda e: e.get(key, 0), reverse=desc)


def compute_kpis(data: list[dict]) -> list[dict]:
    if not data:
        return []
    kpis = []
    # Total count
    kpis.append({"label": "Total Listings", "value": f"{len(data):,}", "sub": store.source_name})
    # Numeric columns summary
    num_cols = [col for col, t in store.schema.items() if isinstance(t, str) and t in ("integer", "float")]
    for col in num_cols[:3]:
        vals = [_num(r.get(col)) for r in data if r.get(col) is not None]
        if vals:
            avg = sum(vals) / len(vals)
            if col in ("price",):
                kpis.append({"label": f"Avg {col.title()}", "value": f"£{avg:,.0f}", "sub": f"Range: £{min(vals):,.0f}–£{max(vals):,.0f}"})
            elif col in ("mileage",):
                kpis.append({"label": "Avg Mileage", "value": f"{avg:,.0f} mi", "sub": f"Max: {max(vals):,.0f} mi"})
            elif col in ("mpg",):
                kpis.append({"label": "Avg MPG", "value": f"{avg:.1f}", "sub": f"Best: {max(vals):.1f} MPG"})
    return kpis[:4]


# ─── LLM query parser ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data analyst AI that converts natural language questions into dashboard specifications.
You have access to a dataset described below. You must generate a JSON response that specifies:
1. A list of chart specifications (1–3 charts)
2. A short insight summary
3. KPI metrics to display

{schema}

Return ONLY valid JSON (no markdown, no explanation) in this exact format:
{{
  "insight": "One concise sentence summarizing the key finding",
  "kpis": [
    {{"label": "string", "value": "string (pre-computed if possible)", "sub": "string"}}
  ],
  "charts": [
    {{
      "id": "chart1",
      "title": "string",
      "type": "bar" | "line" | "pie" | "doughnut" | "scatter" | "horizontalBar",
      "wide": true | false,
      "groupBy": "column_name",
      "secondGroupBy": "column_name" | null,
      "metric": "count" | "avg:column" | "sum:column" | "max:column",
      "xField": "column_name (for scatter only)",
      "yField": "column_name (for scatter only)",
      "colorField": "column_name (for scatter only)",
      "filters": [{{"field": "col", "op": "eq"|"gte"|"lte"|"in", "value": "val"}}],
      "topN": 10 | null,
      "sortDesc": true | false,
      "description": "What this chart shows"
    }}
  ],
  "cannotAnswer": false,
  "cannotAnswerReason": ""
}}

Chart selection rules:
- Time series (year/month) → "line", wide: true
- Comparisons across many categories → "bar" or "horizontalBar"
- Parts of a whole (≤6 categories) → "pie" or "doughnut"
- Two numeric variables → "scatter"
- 2+ groups stacked/grouped → use secondGroupBy
- Set wide: true for line charts and scatter plots

If the question cannot be answered from the available data, set cannotAnswer: true and explain why.
Never invent data or metrics not in the schema. Be precise about column names."""


def _extract_json_block(text: str) -> str:
    """Extract the first top-level JSON object from an LLM response."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return text[start:]


def _sanitize_json_control_chars(text: str) -> str:
    """Replace illegal raw control chars inside JSON strings with escaped forms."""
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ord(ch) < 32:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(" ")
                continue
            out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_string = True
    return "".join(out)


def _parse_llm_json(raw_text: str) -> dict:
    """Parse and lightly repair model output into JSON object."""
    candidate = re.sub(r"```json|```", "", raw_text).strip()
    candidate = _extract_json_block(candidate).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = _sanitize_json_control_chars(candidate)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        return json.loads(repaired)


class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None  # previous dashboard context for follow-ups


@app.post("/api/query")
async def run_query(req: QueryRequest):
    schema_desc = store.schema_summary()
    system = SYSTEM_PROMPT.format(schema=schema_desc)
    client = get_llm_client()

    messages = [{"role": "user", "content": req.query}]
    if req.context:
        messages = [
            {"role": "user", "content": f"Previous dashboard context: {req.context}\n\nFollow-up question: {req.query}"}
        ]

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system},
                *messages,
            ],
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            raise HTTPException(status_code=502, detail="LLM returned an empty response.")
        spec = _parse_llm_json(raw)
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if spec.get("cannotAnswer"):
        return {
            "cannotAnswer": True,
            "reason": spec.get("cannotAnswerReason", "This query cannot be answered from the available data."),
        }

    # Execute each chart spec against real data
    chart_results = []
    for chart_spec in spec.get("charts", []):
        try:
            result = execute_spec(chart_spec, store.data)
            chart_results.append({
                **chart_spec,
                "result": result,
            })
        except Exception as e:
            chart_results.append({**chart_spec, "error": str(e)})

    # Compute KPIs from data if not provided by LLM
    kpis = spec.get("kpis") or compute_kpis(store.data)

    return {
        "cannotAnswer": False,
        "insight": spec.get("insight", ""),
        "kpis": kpis,
        "charts": chart_results,
        "query": req.query,
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = (await file.read()).decode("utf-8-sig", errors="replace")
    try:
        count = store.load_csv(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {e}") from e
    return {"ok": True, "rows": count, "schema": store.schema_summary(), "filename": file.filename}


@app.get("/api/stats")
async def get_stats():
    """Return pre-computed summary stats for initial dashboard render."""
    data = store.data
    if not data:
        return {}

    num_cols = [col for col, t in store.schema.items() if isinstance(t, str) and t in ("integer", "float")]
    cat_cols = [col for col, t in store.schema.items() if isinstance(t, dict)]

    result = {
        "kpis": compute_kpis(data),
        "schema": store.schema_summary(),
        "sourceName": store.source_name,
        "totalRows": len(data),
    }

    # Auto-generate a starter chart for each categorical column
    starter_charts = []
    for col in cat_cols[:4]:
        groups = Counter(str(r.get(col, "")).strip() for r in data)
        top = groups.most_common(10)
        starter_charts.append({
            "id": f"auto_{col}",
            "title": f"Distribution by {col}",
            "type": "bar" if len(top) > 5 else "doughnut",
            "labels": [k for k, _ in top],
            "values": [v for _, v in top],
            "wide": False,
        })

    # Time series if year column exists
    if "year" in store.schema:
        by_year = defaultdict(int)
        for r in data:
            y = r.get("year")
            if y and int(y) >= 2000:
                by_year[int(y)] += 1
        years = sorted(by_year.keys())
        starter_charts.append({
            "id": "auto_year",
            "title": "Inventory count by year",
            "type": "line",
            "labels": [str(y) for y in years],
            "values": [by_year[y] for y in years],
            "wide": True,
        })

    result["starterCharts"] = starter_charts
    return result


@app.get("/api/health")
async def health():
    return {"status": "ok", "rows": len(store.data), "source": store.source_name}


# Serve frontend for both layouts:
# 1) monorepo style: ../frontend/index.html
# 2) single-folder style: ./index.html
FRONTEND_DIR = BASE_DIR.parent / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
INDEX_CANDIDATES = [FRONTEND_DIR / "index.html", BASE_DIR / "index.html"]

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    for index_file in INDEX_CANDIDATES:
        if index_file.exists():
            return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")
