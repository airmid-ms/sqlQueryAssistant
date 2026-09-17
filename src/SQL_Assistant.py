"""
PostgreSQL Query Assistant
------------------------------------------
Connects to a PostgreSQL database and runs a 3-step pipeline before any
query touches your data:
 
  1. Generate - one model call drafts SQL from the plain-English question.
  2. Review - a seperate model call, acting purely as a critic, and may lightly correct the previous query.
  3. Execute - only the reviewed/approved query is run under a read-only Postgres role.

See README.md for setup and usage instructions.
 

"""

import json
import os
import psycopg2
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv() # read local .env file into the environment if present.

client = Anthropic()
model = "claude-sonnet-4-5"

pg_config = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": os.environ.get("PG_PORT", "5432"),
    "dbname": os.environ.get("PG_DBNAME"),
    "user": os.environ.get("PG_USER"),
    "password": os.environ.get("PG_PASSWORD"),
}

# Optional: leave as "None" to pull all tables or limit schema discovery to specific tables.
table_allowlist = None  # e.g. ["sales", "customers", "products"]


# 1. Auto-Discover the Schema.

def get_connection():
    return psycopg2.connect(**PG_CONFIG)
 
 
def discover_schema():
    """Query Postgres's catalog to build a schema description."""
    query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()
 
    tables = {}
    for table_name, column_name, data_type in rows:
        if table_allowlist and table_name not in table_allowlist:
            continue
        tables.setdefault(table_name, []).append(f"{column_name} ({data_type})")
 
    lines = []
    for table, columns in tables.items():
        lines.append(f"Table: {table}")
        for col in columns:
            lines.append(f"  - {col}")
    return "\n".join(lines)
 
 
# 1a. Prompt Setup

def build_system_prompt(schema_description):
    return f"""You are a SQL assistant for a data analyst. You write PostgreSQL
queries against the following schema:
 
{schema_description}
 
Rules:
- Only ever write SELECT queries. Never write INSERT, UPDATE, DELETE, DROP, ALTER,
  TRUNCATE, GRANT, or any statement that modifies data or schema.
- Use standard PostgreSQL syntax.
- If a relative date range is mentioned (e.g. "last quarter"), resolve it using
  CURRENT_DATE in the SQL itself rather than guessing a hardcoded date.
- If the question is ambiguous, make the most reasonable assumption and dont ask follow up questions. 
- Respond with ONLY a JSON object, no markdown fences, no other text:
  {{
    "sql": "<the SQL query>",
    "assumption": "<any assumption you made, or empty string if none>"
  }}
"""

# further feature imvolve asking set follow up questions for certain issues.
# should i prevent SELECT * queries too?

 
 
def generate_sql(question, system_prompt):
    response = client.messages.create(
        model = MODEL,
        max_tokens = 400,
        system = system_prompt,
        messages = [{"role": "user", "content": question}],
    )
    raw = response.content[0].text.strip()
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)
 
 
# 2. Fresh Model Acting as a Review.

def build_review_prompt(schema_description):
    return f"""You are a meticulous senior data analyst reviewing a SQL query
before it is run against a PostgreSQL database. Schema:
 
{schema_description}
 
Check the query for:
1. Safety: it must be a read-only SELECT query. Any write, DDL, or
   permission-altering statement must be rejected.
2. Correctness: does it actually answer the stated question? Check joins,
   date range boundaries, NULL handling, and aggregation grouping.
3. Efficienct: flag anything obviously wasteful (e.g. SELECT * on a wide
   table), but do not block on this alone.
 
You may lightly rewrite the query to fix a real problem (e.g. tightening a
date boundary, fixing a join).

Respond with ONLY a JSON object, no other text:
{{
  "approved": true or false,
  "final_sql": "<the query to run - corrected if you made a fix, or unchanged if it was already correct>",
  "review_notes": "<1-2 sentences: what you checked and any issue found/fixed, or 'looks correct' if nothing to flag>"
}}
 
Set "approved" to false ONLY if the query is unsafe (not read-only) or you are not
confident it answers the question even after attempting a fix.
"""
 
 
def review_sql(question, draft_sql, schema_description):
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=build_review_prompt(schema_description),
        messages=[{
            "role": "user",
            "content": f"Original question: {question}\n\nDraft query:\n{draft_sql}",
        }],
    )
    raw = response.content[0].text.strip()
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)
 
 
# 2a. Safety Guardrail

def is_safe_select(sql):
    """First layer of defense - checks for unwanted key words."""
    normalized = sql.strip().lower()
    if not normalized.startswith("select") and not normalized.startswith("with"):
        return False
    forbidden = [
        "insert", "update", "delete", "drop", "alter", "create",
        "truncate", "grant", "revoke", "attach", ";--",
    ]
    return not any(word in normalized for word in forbidden)
 
 
# 4. Execute & Format

def run_query(sql):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    conn.close()
    return columns, rows
 
 
def format_results(columns, rows):
    if not rows:
        return "No results."
    widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0)) for i, c in enumerate(columns)]
    header = " | ".join(c.ljust(w) for c, w in zip(columns, widths))
    divider = "-+-".join("-" * w for w in widths)
    body = "\n".join(
        " | ".join(str(val).ljust(w) for val, w in zip(row, widths)) for row in rows
    )
    return f"{header}\n{divider}\n{body}"
 
 
def summarize_answer(question, columns, rows):
    preview = rows[:20]
    response = client.messages.create(
        model = MODEL,
        max_tokens = 150,
        messages = [{
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Columns: {columns}\n"
                f"Result rows (sample): {preview}\n\n"
                "Give a single, concise plain-English sentence answering the question "
                "based on this data. No preamble."
            ),
        }],
    )
    return response.content[0].text.strip()
 
 
# 5. Pipeline Loop

def run_assistant():
    print("=== SQL Query Assistant (PostgreSQL) ===")
    print("Discovering schema...")
    schema_description = discover_schema()
    if not schema_description:
        print("No tables found in the 'public' schema. Check your connection/permissions.")
        return
    print(f"Found schema:\n{schema_description}\n")
 
    system_prompt = build_system_prompt(schema_description)
 
    print("Type your query. Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "quit":
            break
 
        try:
            draft = generate_sql(question, system_prompt)
        except json.JSONDecodeError:
            print("Couldn't parse the model's response, try rephrasing.\n")
            continue
 
        draft_sql = draft["sql"]
        print(f"\nDraft SQL:\n  {draft_sql}")
        if draft.get("assumption"):
            print(f"(Assumption: {draft['assumption']})")
 
        # Review step: a second call scrutinizes the draft before execution
        try:
            review = review_sql(question, draft_sql, schema_description)
        except json.JSONDecodeError:
            print("Couldn't parse the reviewer's response - skipping this query.\n")
            continue
 
        print(f"\nReview: {review['review_notes']}")
 
        if not review.get("approved", False):
            print("Blocked: reviewer did not approve this query.\n")
            continue
 
        sql = review["final_sql"]
        if sql != draft_sql:
            print(f"Reviewer revised the query:\n  {sql}")
 
        # --- Code-level guardrail, independent of what the reviewer says ---
        if not is_safe_select(sql):
            print("Blocked: query failed the read-only safety check.\n")
            continue
 
        try:
            columns, rows = run_query(sql)
        except psycopg2.Error as e:
            print(f"SQL error: {e}\n")
            continue
 
        print("\nResults:")
        print(format_results(columns, rows))
 
        answer = summarize_answer(question, columns, rows)
        print(f"\nAnswer: {answer}\n")
 
 
if __name__ == "__main__":
    missing = [k for k in ("PG_DBNAME", "PG_USER", "PG_PASSWORD") if not os.environ.get(k)]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set your ANTHROPIC_API_KEY environment variable first.")
    elif missing:
        print(f"Missing environment variables: {', '.join(missing)}")
    else:
        run_assistant()
