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

## Running locally

Graph has been run inpa4.ipynyb, as outputs are not added in databricks commit, pics and references pasted below:


> **Example of the level of detail expected** (replace with your own steps/values):
>
> 1. **Ingest the corpus** (run once, from a Databricks notebook):
>    ```python
>    from rag.ingest import ingest
>    ingest(spark, volume_path="/Volumes/main/default/pa4/annual_report.pdf")
>    ```
>    This parses the PDF, chunks it into `main.default.ali_analyst_chunks`, and syncs the
>    Vector Search index `main.default.ali_analyst_index`. Wait until the index is `READY`.
>
> 2. **Build and run the graph** in `pa4.ipynb`:
>    ```python
>    from agent.graph import build_graph
>    graph = build_graph()          # uses config.py + rag/store.py + the MCP server
>    result = graph.invoke({"messages": [{"role": "user",
>              "content": "What was the net revenue in 2023?"}]})
>    print(result["messages"][-1].content)
>    ```
>
> 3. **Test queries I ran** (retrieval-only, computation-only, combined):
>    | Query | Answer produced |
>    |-------|-----------------|
>    | "What was the net income in 2023?" | ¥1.11 trillion [source: annual_report.pdf, p.4] |
>    | "What is 15% of 2.4 billion?" | 360 million |
>    | "What was 2023 revenue, and its value after 10% growth?" | ¥16.91T → ¥18.60T (16.91 × 1.10) |


## Deployment

TODO: how you logged, registered, and served the model; endpoint name; URL.

## Design decisions

TODO: graph architecture, routing, deployment choices.

---

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
   - TODO
2. What happens to in-flight requests when you deploy a new model version to the same endpoint? How does Databricks handle the transition?
   - TODO

### Task 3.2 — Client
1. Why is exponential backoff better than fixed-interval retries for a model serving endpoint?
   - TODO
2. Your client has a `max_retries` parameter. What is the danger of setting it too high in a production system with many concurrent users?
   - TODO
3. When would you choose `ask_streaming()` over `ask()`? Give a concrete UX example.
   - TODO

### Bonus A — CI/CD (if attempted)
1. Why should the deploy step only run on `main` and not on feature branches?
   - TODO
2. What would you add to this pipeline to prevent deploying a model that performs worse than the current version? Describe the gate.
   - TODO

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
