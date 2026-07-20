# CS4603 PA4 — Document Analyst (Student Submission)

> This is your **submission file**. `README.md` is the assignment spec — this document is where you write up your work.
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment spec.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```
I configured the following variables in `.env` for local development. The actual token is private and is not committed.

```dotenv
DATABRICKS_HOST=https://dbc-005bf88c-4348.cloud.databricks.com
DATABRICKS_TOKEN=<private PAT>
DATABRICKS_MODEL=databricks-meta-llama-3-3-70b-instruct
EMBEDDINGS_ENDPOINT=databricks-gte-large-en

UC_CATALOG=cs4603
UC_SCHEMA=default
UC_MODEL_NAME=s27100380_document_analyst

SOURCE_TABLE=cs4603.default.s27100380_analyst_chunks
VECTOR_SEARCH_ENDPOINT=s27100380-vs-endpoint
VECTOR_SEARCH_INDEX=cs4603.default.s27100380_analyst_index

SERVING_ENDPOINT_NAME=s27100380-document-analyst
SECRET_SCOPE=cs4603-deploy
```

The source PDF was uploaded to this Unity Catalog volume path:

```text
/Volumes/cs4603/default/pa4/annual_report.pdf
```

I used a Databricks PAT during development. The token was stored in `.env` locally and in the `cs4603-deploy` Databricks secret scope for serving; it was never placed directly in the model artifact or committed files.

## Corpus ingestion and Vector Search

From `pa4.ipynb` on Databricks, I set the environment values and built the Delta chunks table from the uploaded report:

```python
from rag.ingest import build_chunks_table

build_chunks_table(
    spark,
    volume_path="/Volumes/cs4603/default/pa4/annual_report.pdf",
    chunks_table="cs4603.default.s27100380_analyst_chunks",
)
```

The final ingestion output was:
```text
Parsed 1 document into cs4603.default.s27100380_analyst_chunks_parsed_documents.
Created 7 unique chunks in cs4603.default.s27100380_analyst_chunks.
```

The source Delta table uses Change Data Feed and feeds a triggered Delta Sync Vector Search index:

```text
Vector Search endpoint: s27100380-vs-endpoint
Vector Search index:    cs4603.default.s27100380_analyst_index
Embedding endpoint:     databricks-gte-large-en
Final index state:      ONLINE/READY
```

`rag/store.py` creates a `DatabricksVectorSearch` retriever over this managed index and requests citation metadata (`chunk_id`, `source`, and `page`). The managed index already has `chunk_to_retrieve` configured as its source column, so the client does not pass a separate `text_column`. This same retriever factory is used by both the notebook graph and the deployed model.

## Running the graph

I built the complete graph in `pa4.ipynb`:

```python
from agent.graph import build_graph

graph = build_graph()

result = graph.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What was the revenue in 2023?",
        }
    ]
})


print(result["messages"][-1].content)
```

The compiled graph was visualized in the notebook. Its control flow was:

```text
START
  -> planner
  -> supervisor
       -> rag_agent -> supervisor
       -> mcp_tools -> supervisor
       -> synthesizer
  -> END
```
### Final local test results

| Query | Final local result |
|---|---|
| What was the net income in 2023? | The graph retrieved FY2023 revenue of ¥16.91 trillion and calculated expenses of ¥15,786 billion, producing ¥1,124 billion, with evidence from `annual_report.pdf`, page 2.0. |
| What is 15% of 2.4 billion? | 3.6e+08, or 360 million. |
| What was the revenue in 2023, and what would a 10% increase look like? | Revenue was ¥16.91 trillion; the increase was ¥1.691 trillion; the resulting total was ¥18.601 trillion. |

For the combined query, the RAG specialist first retrieved the revenue and citation. The MCP calculation steps then used earlier values from `step_results` to calculate the increase and final total. The synthesizer combined the step-labelled results into one response.

### Offline smoke test

```bash
PYTHONPYCACHEPREFIX=/tmp/pa4_pycache \
python -m pytest tests/test_smoke.py -q \
  -o cache_dir=/tmp/pa4_pytest_cache
```

The final result was:

```text
..                                                                       [100%]
```

## Deployment

### Model definition and registration

`deployment/agent_model.py` is a models-from-code definition. At import time it validates all required environment variables, constructs the Databricks-hosted LLM, creates the shared Vector Search retriever, loads the bundled stdio MCP tools, builds the graph, and finishes with:

```python
mlflow.models.set_model(graph)
```

`deployment/deploy.py` logs the graph with `mlflow.langchain.log_model`. I used the raw LangGraph-state interface rather than a `ChatModel` wrapper. The model was logged with an explicit input/output signature and these code paths so the serving container could import the complete application:


```text
agent/
rag/
tools/
config.py
```

The model was registered in Unity Catalog as:

```text
cs4603.default.s27100380_document_analyst
```

The final working registered version was version `3`.

Two serving-specific issues were resolved before version 3:

1. MLflow places bundled `code_paths` under `/model/code`, so the MCP server path is resolved as `/model/code/tools/mcp_server.py` inside the container while retaining the repository-relative path for local execution.
2. Databricks Model Serving replaces `sys.stderr` with a logging proxy without `fileno()`. The bundled stdio MCP subprocess requires an OS-backed error stream, so the serving model redirects MCP subprocess errors to `os.devnull` through the MCP stdio client. The supplied `tools/mcp_server.py` itself was not modified.

### Serving endpoint

The endpoint was created or updated with the Databricks Python SDK using `EndpointCoreConfigInput` and `ServedEntityInput`:

```text
Endpoint name:  s27100380-document-analyst
Model:          cs4603.default.s27100380_document_analyst
Version:        3
Workload size:  Small
Scale to zero:  enabled
Final state:    READY
```

Endpoint page:

```text
https://dbc-005bf88c-4348.cloud.databricks.com/ml/endpoints/s27100380-document-analyst
```

Because the graph was logged directly with `mlflow.langchain.log_model`, the response is a batch list containing raw LangGraph state. I parse the answer with:

```python
data[0]["messages"][-1]["content"]
``


### Deployed endpoint results

| Query | Local vs. deployed | Result |
|---|---:|---|
| What was the net income in 2023? | Not exactly identical | Both produced ¥1,124 billion from ¥16.91 trillion revenue and ¥15,786 billion expenses, with the same annual-report evidence. The deployed answer contained more explicit unit conversion and calculation wording. |
| What is 15% of 2.4 billion? | Not exactly identical | Both produced 3.6e+08, or 360 million. Only the phrasing differed. |
| What was the revenue in 2023, and what would a 10% increase look like? | Exactly identical | Both returned ¥16.91 trillion, a ¥1.691 trillion increase, and a ¥18.601 trillion total. |

Strict string equality was `False`, `False`, and `True`, respectively. The first two answers were semantically consistent despite small differences in wording. Exact text is not guaranteed because the model runs in separate local and serving executions, and minor generation or retrieval-order differences can change phrasing even at temperature zero.

### Cold and warm latency

The measured endpoint latency was:

```text
Cold/first request: 52.19 seconds
Warm request:        8.73 seconds
Difference:         43.46 seconds
```

## Design decisions

### State and sequential execution

`AnalystState` is a `TypedDict` containing conversation messages, the ordered plan, the current step index, accumulated step results, the supervisor route, and the final answer. The `messages` channel uses LangGraph's `add_messages` reducer so new messages are appended instead of replacing history.

The planner produces two to five atomic steps when valid JSON is returned and falls back to the original query as one step on a parse failure. Each specialist appends one result and increments `current_step_index`. This makes execution order explicit and lets later MCP calculation steps receive previous results as context.

### Deployment choice

I selected the simpler raw-state `mlflow.langchain.log_model` approach. It preserves the graph's internal fields for inspection and directly supports the compiled LangGraph object. The tradeoff is that callers must parse the raw batch state rather than use `response.choices[0]`. The model queries managed Vector Search at inference time

## Analysis Questions

> Answer in your own words. Each question is copied from the assignment so you don't have to flip back.

### Task 1.2 — Planner
1. What happens when the planner produces steps that depend on each other (e.g., step 3 needs the result of step 1)? How does your architecture handle this?
   - The graph executes planner steps sequentially using current_step_index, each specilaist appends its output to step_results, and so a later step can acvcess and use data produced by an earlier step. eg. rag agent retrieves f72023 revenue-> mcp node used val to calculate growth. Handling in architecture: the planner should make dependencies eplicit by stating : "Using ans from {step} in {next_step} , calculating..." . and the executor node needs to have the prev step_results in context for the current step_result iteration
2. Would a replanning step after each execution improve or hurt performance for this use case? Justify with an example.
   - Replanning would hurt performance as it would add one LLM invoke call after each execution. increasing latency, cost, for not all necessary replanning. eg. once revenue retrieved, graph can immediately move onto calculating growth via next node without requiring LLM's reasoning in the middle. adding edges would aldo increase complexity and not allow for modular workflow

### Task 1.3 — Supervisor
1. Your supervisor makes a routing decision per step. What is the failure mode if it misroutes? How would you detect and recover from a misroute?
    - A misroute sends a step requent to a specialist that isnt designed for the task/ cannot do it. for eg, it sends document_loop to 'mcp_tools' could cause a failed tool call. This would be detected via the invalid tool call, empty retrieval results and tracing step answers till the problem node/tool call. theshpervisor will normalizeoutput on its end and send a fallback alert when output is invalid. 

2. Compare this supervisor pattern with a single ReAct agent that has access to all tools. When is the supervisor pattern worth the added complexity?
   -  A single ReAct agent is simpler because one model decides which tool to call while reasoning through the entire request  however its behavior can  be less predictable when it has many tools or when a query requires multi step reasoning and different complex tasks.  The supervisor adds routing logic and additional LLM calls, but gives each specialist a focused responsibility and makes the execution path easier to debug and verify, the extra complexity is valid for a multi agent as querires need both lookup and computation.
   

### Task 1.4 — RAG Agent
1. The RAG agent retrieves for a single decomposed step, not the full user query. How does this affect retrieval quality compared to retrieving for the original question?
   - This improves precision as the query focuses on one single fact rather than doing both retrieval and calculation requirements of the whole query.  for a q like "find fy2023 revenue" is better than "find fy2023 revenue and the growth calculation" for getting a precise accurate answer. this way by modularizing rag retrieval in multi agent reasoning the LLM can handle more complex queries.
2. If the planner produces a vague step like "find relevant financial data," how would you improve the retrieval query before sending it to the vector store?
   - Would rewrite step using more context from og question and previous steps/nodes. the addedinfo would include metrics, company info, units and doc metadata. would also add a specififc return type eg. return as an annual report pdf file ,focused specifically on the years []. this clarifies the input to llm

### Task 2.1 — Model Definition
1. Why does `models-from-code` require a self-contained file? What breaks if you reference external state (e.g., a database running only on your laptop)?
   - MLFlow stores model definition source and re excutes it inside a container, the container does not inherit the notebook variables or python objects or processes. therefore the model file must recreate the graph from importable code. if it references  external state such as a local database , the model may failduring endpoint startup or inference because the resource does not existin the serving environment of the model.
2. Your model calls a managed Vector Search index at inference time rather than embedding documents into the container image. What are the tradeoffs (freshness, cold-start size, latency, failure modes) of querying an external index vs. baking the corpus into the model artifact?
   - using vector search index keeps the model artifact smaller, and allows document updates via index synchronization. the trandeoff is the num of of network calls and complexity, as every retrieval now has a network call and is dependent on the index, authentication and availability of the database.

### Task 2.3 — Serving Endpoint
1. Why must you pass `DATABRICKS_TOKEN` as an environment variable to the endpoint, even though it's already authenticated to serve models?
   - T
2. What happens to in-flight requests when you deploy a new model version to the same endpoint? How does Databricks handle the transition?
   - TODO

### Task 3.2 — Client
1. Why is exponential backoff better than fixed-interval retries for a model serving endpoint?
   - Exponential backoff increases the delay after every failure, reducing request pressure and giving the model endpoint more time to configure and become availavable after the failure. fixed interval retries hit the failure repeatedly even before it 
   has recovered. a 503/429 error usually means endpoint is overloaded, so exponential backoff works better.
2. Your client has a `max_retries` parameter. What is the danger of setting it too high in a production system with many concurrent users?
   - with concurrency excessive retries can create a situation where each failed request produces additional requests, and multiple failures at a time overbuden the endpoint, increasing latency and making recovery lengthy.
3. When would you choose `ask_streaming()` over `ask()`? Give a concrete UX example.
   - streaming is useful in a interative chat software when the answer needs to be shown progressively. for eg ask is more appropriate when a caller needs one answer before the next, in a calculation. for a financial analysis app, this would be any point where LLM tool is performing calculations. otherwise, a scenerio where user wants a summary of a financial report, ask streaming is better as it shows the beginning text before it prints the final citatons etc, giving user a timely answer where user can see the thinking of the llm too.

### Bonus A — CI/CD (if attempted)
## Execution evidence

The GitHub Actions workflow completed successfully: lint passed, the offline
smoke test passed, a new model version was registered, and the serving endpoint
reached `READY`.

![Successful Bonus A workflow](evidence/bonusA.png)

1. Why should the deploy step only run on `main` and not on feature branches?
   - main is the reviewed source of code, feature branches may contrain incomplete or experimental data or conflicting changes, deplying from there could cause errors and allow different branches to run over each others deplyments. 
2. What would you add to this pipeline to prevent deploying a model that performs worse than the current version? Describe the gate.
   - add an evaluation job in between testing and deployment that compares new model to current one. would run retrieval, calculation, combined, empty retrieval and citation queries and evaluation on precision correctness . using mflow. if model below threshold after comparison then it would not be deployed

### Bonus B — `databricks-agents` SDK (if attempted)
1. Compare the `agents.deploy()` approach with the manual MLflow + CLI approach from Part 2. What control do you gain or lose with each?
   - TODO
2. The Review App enables human feedback collection. How would you use this feedback to improve the agent over time? Describe a concrete feedback loop.
   - TODO

### Bonus C — Standalone MCP server (if attempted)
1. You moved the MCP server out of the model container. What did you gain (scaling, deployment, security, observability) and what new failure modes did you introduce (network, auth, latency, availability)?
   - TODO
2. The remote MCP server now needs its own authentication. How would you secure it so that only your serving endpoint — not the public internet — can call the tools?
   - TODO
3. When is bundling the tools in the container (Part 1) the *better* choice, and when is a separately deployed tool service (Bonus C) worth the extra moving parts?
   - TODO
