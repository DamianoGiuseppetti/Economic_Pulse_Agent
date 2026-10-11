"""
Economic Pulse Agent: local Streamlit chat. Run with:  streamlit run app.py

Local only. It calls the Anthropic and FRED APIs with the keys in your own environment (.env), so the
model cost is yours. Under each answer an expander shows the tool calls, tokens, cost, iterations,
grounding status and prompt-cache status. When a call would be expensive (see approval.py) the answer
pauses and asks you to approve or decline it.
"""

import os
import time

import streamlit as st

import config
from ui_support import UIApprover, start_job, turn_details

st.set_page_config(page_title="Economic Pulse Agent", page_icon="📈")


def _new_agent():
    from agent import EconomicPulseAgent
    approver = UIApprover()
    return EconomicPulseAgent(approve=approver), approver


def _keys_missing() -> list[str]:
    missing = []
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        missing.append("ANTHROPIC_API_KEY")
    if not config.FRED_API_KEY:
        missing.append("FRED_API_KEY")
    return missing


state = st.session_state
if "agent" not in state:                      # tests preset agent and approver
    missing = _keys_missing()
    if missing:
        st.error(f"Missing {', '.join(missing)}. Put them in a .env file next to this app "
                 "(see .env.example) or export them, then reload.")
        st.stop()
    state.agent, state.approver = _new_agent()
state.setdefault("messages", [])
state.setdefault("job", None)

st.title("Economic Pulse Agent")
st.caption("US macro and market data from FRED and Yahoo Finance. Runs on your machine with your own API keys; "
           "not financial advice.")

with st.sidebar:
    st.write(f"Session `{state.agent.session_id}`")
    if st.button("New conversation", disabled=state.job is not None):
        state.agent, state.approver = _new_agent()
        state.messages = []
        st.rerun()
    st.caption("Tools are read-only. Approval is asked only for expensive calls: a large uncached download, "
               "many tool calls in one question, or a session that has already cost a lot.")


def show_details(details: dict) -> None:
    with st.expander("Details: tool calls, tokens, cost"):
        cols = st.columns(3)
        cols[0].metric("Cost", f"${details['cost_usd']:.4f}")
        cols[1].metric("Iterations", details["iterations"])
        cols[2].metric("Time", f"{details['seconds']} s")
        st.write(f"Tokens: **{details['tokens_in']:,}** in / **{details['tokens_out']:,}** out")
        st.write(f"Grounding check: **{details['grounding']}** · Prompt cache: **{details['prompt_cache']}**")
        if details["tool_calls"]:
            st.dataframe(
                [{"tool": c["tool"], "arguments": str(c["arguments"]), "status": c["status"], "data": c["data"],
                  "ms": c["latency_ms"]} for c in details["tool_calls"]],
                hide_index=True, use_container_width=True)
        else:
            st.write("No tools were called.")


for message in state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details"):
            show_details(message["details"])

question = st.chat_input("Ask about the US economy or markets", disabled=state.job is not None)
if question and state.job is None:
    state.messages.append({"role": "user", "content": question})
    state.job = start_job(state.agent, question)
    with st.chat_message("user"):
        st.markdown(question)

job = state.job
if job is not None:
    with st.chat_message("assistant"):
        # Wait for the answer, or for the approval gate to ask something.
        deadline = time.monotonic() + 600
        while job.running and state.approver.pending is None and time.monotonic() < deadline:
            time.sleep(0.15)
        request = state.approver.pending
        if request is not None:
            st.warning(f"**Approval needed** to run `{request.tool}`: {request.reason}.")
            st.json(request.args, expanded=False)
            left, right, _ = st.columns([1, 1, 4])
            left.button("Approve", type="primary", on_click=state.approver.decide, args=(True,), key="approve")
            right.button("Decline", on_click=state.approver.decide, args=(False,), key="decline")
            st.stop()                       # the click reruns the script, which resumes polling
        if job.running:                     # still working after the wait: keep polling on the next run
            st.rerun()
        job.thread.join(timeout=5)
    if job.result is not None:
        state.messages.append({"role": "assistant", "content": job.result.answer, "details": turn_details(job.result)})
    elif job.error is not None:
        state.messages.append({"role": "assistant", "content": f"Something went wrong: {type(job.error).__name__}: {job.error}"})
    state.job = None
    st.rerun()
