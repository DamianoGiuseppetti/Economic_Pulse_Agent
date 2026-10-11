# Economic Pulse Agent

Economic Pulse Agent answers questions about US macro data and markets in plain language. You ask "What is the current unemployment rate?" or "Has anything unusual happened in CPI over the past year?", and a Claude Haiku 4.5 agent decides which tools to call, fetches the data from FRED and Yahoo Finance, and writes the answer from what came back.

Two rules shape it. The model never does the arithmetic or decides what counts as unusual: direction, percentage changes, year-over-year inflation and anomaly scores are computed in Python and quoted. And any cause the model offers has to appear in a tool result or carry the label "(not verified by my data)". A rule-based check enforces that after every answer.

## Example runs

Real runs from 2026-10-07 with `claude-haiku-4-5`, copied without edits. I checked every number against FRED with [`docs/verify_examples.py`](docs/verify_examples.py). More are in [`economic_pulse_agent/examples.txt`](economic_pulse_agent/examples.txt).

### 1

> **You:** What is the current unemployment rate?
>
> **Agent:** The current unemployment rate is **4.2%** as of September 2026 (published October 2, 2026). This represents an increase of 0.1 percentage points from August's 4.1%.
>
> The next employment report will be released on November 6, 2026.

### 2

> **You:** Has anything unusual happened in CPI over the past year?
>
> **Agent:** Over the past year, 2 of 10 CPI observations were unusual:
>
> **June 2026**: CPI fell 0.42% month-over-month, a notably large decline (z = –3.49, 74.8th percentile). This was the biggest drop relative to what's typical for CPI month-to-month changes.
>
> **March 2026**: CPI rose 0.87% month-over-month, an unusually large jump (z = 3.21, 96.6th percentile), among the largest monthly increases in the past decade.
>
> The most recent observation (August 2026, published September 11) showed CPI up 0.4% from the previous month, which was within normal range. The next CPI release is scheduled for October 14.
>
> These moves suggest there was some volatility in price pressures in the first half of 2026, with a sharp acceleration in March followed by moderation by June, though neither move was truly extreme in absolute terms—they were just notably larger than CPI's typical month-to-month swings over the past decade.

### 3

> **You:** What was the unemployment rate between January 2025 and August 2026?
>
> **Agent:** The unemployment rate between January 2025 and August 2026 rose gradually from 4.0% to 4.1%, an increase of 0.1 percentage points. Here are the monthly values:
>
> **January 2025 – August 2026 Unemployment Rate:**
> - Started at **4.0%** (January 2025)
> - Peaked at **4.5%** (November 2025)
> - Ended at **4.1%** (August 2026)
>
> The rate climbed through the first half of 2025, reaching its high of 4.5% in November, then gradually declined through early 2026 before stabilizing around 4.1-4.2% from mid-2026 onward.
>
> The data was released on September 4, 2026. The next unemployment release is scheduled for November 6, 2026.

> **You:** And core CPI over the same period?
>
> **Agent:** Core CPI (excluding food and energy) rose from 324.64 in January 2025 to 337.77 in August 2026, up 4.04%. Year-over-year inflation declined significantly over the period, falling from 3.28% in January 2025 to 2.45% in August 2026.
>
> **Key pattern:** Core inflation peaked at 3.28% in early 2025, then generally trended down through spring 2025, hitting a low of 2.77% in May. It rebounded mid-year (reaching 3.11% in August 2025) before resuming its decline through early 2026. The latest reading of 2.45% (August 2026) continues that downward momentum.
>
> The August 2026 data was released on September 11, 2026. The next CPI release is scheduled for October 14, 2026.

## The Streamlit app

![The Streamlit app answering the CPI question](docs/app.png)

A chat box over the same agent. Under each answer, a details panel shows the cost, model calls, time, tokens, tool calls with their arguments, the grounding check result and whether the prompt cache was hit. When a call would be expensive, the app asks with Approve and Decline buttons before running it. It listens on localhost only.

## Architecture

```mermaid
flowchart LR
    Q([Question]) --> E["AgentExecutor<br/>Claude Haiku 4.5<br/>(async, prompt-cached)"]
    MEM[("Entity memory<br/>per session")] -.->|recently discussed| E
    E -->|tool call| AP{"Approval gate<br/>cost and scope"}
    AP -->|approved| T["Six tools"]
    AP -.->|declined| E
    T --> CACHE[("SQLite cache")]
    T --> RB["Retry + circuit breaker<br/>per source"]
    RB --> SRC["FRED / Yahoo Finance / Fed"]
    T -->|tool results| E
    E --> GR["Grounding check<br/>(unsupported causes)"]
    GR --> A([Answer])
    E --> LOG[("SQLite log<br/>calls, tokens, decisions")]
```

LangChain's `AgentExecutor` runs the loop: the model picks a tool, the tool runs, the result goes back, and this repeats for at most five model calls. Tool results are compact JSON with long series downsampled and the arithmetic already done, and everything the model quotes comes from them.

## The six tools

| Tool | What it does | Series and symbols |
|---|---|---|
| `get_economic_series` | Observations of a FRED series, with direction, % change, year-over-year inflation for price indexes and release dates | Any FRED series; 14 common ones are offered to the model |
| `get_stock_data` | Yahoo Finance price bars, with the largest fall and rise in order | `^GSPC`, `SPY`, `QQQ`, `^IXIC`, `^DJI`, `^VIX`, `TLT`, or any ticker |
| `compare_periods` | One series aggregated over two windows, with the delta, % delta and direction | Any FRED series |
| `fetch_macro_context` | Rule-based sentiment from VIX, the high-yield credit spread and consumer sentiment, plus recent Fed statements | Not news analysis |
| `detect_anomalies` | Scores each observation's change against its own previous ten years and returns z, percentile and `n_flagged` of `n_scored` | `UNRATE`, `FEDFUNDS`, `DGS10`, `DGS2`, `T10Y2Y`, `MORTGAGE30US`, `UMCSENT`, `PAYEMS`, `CPIAUCSL`, `CPILFESL`, `PCEPILFE`, `GDPC1`, `ICSA`, `A191RL1Q225SBEA` |
| `generate_narrative` | A three to five sentence factual brief about one series, assembled in Python from the other tools' results with no model call | The same 14 series |

## How it was built

I built it in five stages: a data layer (SQLite cache over FRED and Yahoo Finance), an agent loop on the raw Anthropic SDK, a port to LangChain's `AgentExecutor`, the anomaly and narrative tools, and finally hardening plus the app. Claude Code did the implementation from written specs, and each stage had to pass its tests before the next began.

Three decisions matter most. Everything numeric is computed in Python: the anomaly tool scores changes rather than levels, uses the median and MAD, and compares each observation only against the ten years before it. Causes are policed by a simple word-list check on every answer. And the agent is hardened the way a deployed one would be: prompt caching, retries and per-source circuit breakers, concurrent tool calls, an approval gate for expensive calls, and per-session memory of the series and date windows already discussed.

## Run it yourself

You need Python 3.10 or newer (I tested on 3.10), a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) and an [Anthropic API key](https://console.anthropic.com/settings/keys). The Anthropic key is billed to you, so set a spending limit in the console first. No keys are in this repository.

```bash
git clone https://github.com/DamianoGiuseppetti/Economic_Pulse_Agent.git
cd Economic_Pulse_Agent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your two keys in .env
cd economic_pulse_agent
```

Chat in the terminal. It asks before an expensive call, and piped input approves automatically:

```bash
python agent.py
```

Run the local web app. Start it from this directory so `.streamlit/config.toml` applies. Streamlit's own default is to listen on every network interface, which would let anyone on your network spend your API credit:

```bash
streamlit run app.py
```

Regenerate `examples.txt` (seven questions, about five cents):

```bash
python agent.py --examples
```

Run the tests:

```bash
python -m pytest test_agent.py -q
```

The suite has three tiers. Most tests drive the agent with a scripted fake model and a bundled FRED snapshot (`fixtures/`), so they need no keys and no network. Seven more fetch live FRED or Yahoo Finance data without calling the model, so they need a FRED key and a connection. With those, 121 tests pass and the 35 live-model tests skip. With an Anthropic key set too, all of them run and call the real model. The data and cache live in `economic_pulse_agent/data/`, which git ignores.

## Costs

Haiku 4.5 costs $1 per million input tokens and $5 per million output tokens, with cached input at a tenth of the input price and a cache write at 1.25 times. I measured the following on 2026-10-07.

| | Result |
|---|---|
| One typical question | $0.003 to $0.011, two model calls, about 4 to 7 s end to end |
| Heaviest question seen (three series in one turn) | $0.020, about 10 s |
| The five original example questions | $0.02 to $0.03 with caching, $0.066 to $0.076 without |
| First question of a cold session | about $0.006 more, for one cache write |
| Tool latency | 0.7 to 1.5 s on a cold fetch, about 5 ms on a cache hit |
| Three-series question, cold caches | 9 to 11 s with concurrent tools, 13 to 18 s sequentially |

I have not totaled what the live-model tests cost, so keep your spending limit in place before running them.

## Known limitations

- On "was X unusual?" the model sometimes also calls `get_economic_series`, a redundant call. Tool routing is probabilistic, and a question like "What is going on with unemployment?" occasionally makes an extra call too.
- The grounding check is a word list. It catches named causes (earnings, tariffs, rate-cut expectations) that no tool result supports, but not general statements about how markets work, which the model occasionally adds to advice-style questions.
- About 36% of FEDFUNDS monthly changes are exactly zero, so any real move scores a very high z. For other series, the z-score and the percentile can disagree in direction, and the model doesn't always explain why.
- The approval thresholds (1,300 observations, four tool calls, $0.15) come from a small sample of my own usage, and normal questions stay well under all three. After a decline, the model may answer from a coarser source and label it without saying you declined.
- Circuit breaker and approval state live in the process and reset on restart. Memory covers one session and nothing carries over to the next.
