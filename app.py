from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml

st.set_page_config(page_title="InstructionDrift | LLM Reliability Lab", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

# ---------- Theme ----------
st.markdown("""
<style>
:root { --ink:#17233f; --muted:#66758e; --line:#e5eaf3; --blue:#3b82f6; }
.block-container {padding-top:1.3rem; padding-bottom:2rem; max-width:1500px;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#111b30 0%,#192641 100%);}
[data-testid="stSidebar"] * {color:#eef3ff !important;}
[data-testid="stMetric"] {background:#fff;border:1px solid #e7edf6;padding:15px 17px;border-radius:15px;box-shadow:0 3px 12px rgba(23,35,63,.035);}
[data-testid="stMetricLabel"] {color:#66758e;}
.panel {background:#fff;border:1px solid #e5eaf3;border-radius:16px;padding:18px 20px;margin-bottom:12px;}
.small-muted {color:#71809a;font-size:.88rem;}
.status-pill {display:inline-block;border-radius:999px;padding:5px 10px;background:#e8f8ef;color:#147d4b;font-size:.8rem;font-weight:600;}
hr {border-color:#e8edf5;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(r"C:\Users\Evanglin Pradeena.R\Downloads\instructiondrift\instructiondrift")
DEFAULT_CONFIG = ROOT / "config.yaml"

# ---------- Safe helpers ----------
def read_yaml(path: Path) -> dict:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def read_text(path: Path, max_chars: int = 30000) -> str:
    try: return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception: return ""

def discover_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not root.exists(): return []
    ignored = {".git", "__pycache__", ".venv", "venv", "node_modules"}
    found=[]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes and not any(part in ignored for part in p.parts):
            found.append(p)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)

def read_table(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() in (".xlsx", ".xls"): return pd.read_excel(path)
        if path.suffix.lower() == ".parquet": return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Couldn't read {path.name}: {e}")
        return pd.DataFrame()

def active_run_dir(cfg: dict) -> Path:
    name = str(cfg.get("run_name", "tiny_pilot"))
    return ROOT / "results" / name

def pretty_num(x: Any, digits: int = 1) -> str:
    try:
        v=float(x)
        if pd.isna(v): return "—"
        return f"{v*100:.{digits}f}%" if abs(v) <= 1 else f"{v:,.{digits}f}"
    except Exception: return "—"

def find_summary(run_dir: Path) -> tuple[Path|None, pd.DataFrame]:
    candidates=[]
    for p in discover_files(run_dir, (".csv", ".xlsx", ".xls")):
        nm=p.stem.lower()
        if "summary" in nm and ("test" in nm or "result" in nm or nm == "summary"):
            candidates.append(p)
    if not candidates:
        candidates=[p for p in discover_files(run_dir, (".csv", ".xlsx", ".xls")) if "summary" in p.stem.lower()]
    for p in candidates:
        df=read_table(p)
        if not df.empty: return p, df
    return None, pd.DataFrame()

def col_name(df: pd.DataFrame, *terms: str) -> str|None:
    for c in df.columns:
        s=str(c).lower().strip()
        if all(t in s for t in terms): return c
    return None

def ollama_api_base(cfg: dict) -> str:
    base=str(cfg.get("base_url", "http://localhost:11434/v1")).rstrip("/")
    if base.endswith("/v1"): base=base[:-3]
    return base

def ollama_models(cfg: dict) -> tuple[bool, list[str], str]:
    base=ollama_api_base(cfg)
    try:
        r=requests.get(base+"/api/tags", timeout=2.5); r.raise_for_status()
        models=[m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
        return True, models, "Connected"
    except Exception as e: return False, [], str(e)

def call_ollama(cfg: dict, model: str, messages: list[dict], temperature: float=0.2) -> str:
    url=ollama_api_base(cfg)+"/api/chat"
    payload={"model":model, "messages":messages, "stream":False, "options":{"temperature":temperature}}
    r=requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return str(r.json().get("message", {}).get("content", "")).strip()

def policy_system_prompt(policy: str, rules: str, user_turn: int, history: list[dict]) -> tuple[str, str]:
    base="You are an assistant in an instruction-following demonstration. Follow the user's active rules unless they conflict with safety or system requirements. Be helpful and accurate."
    active=rules.strip() or "Answer clearly and accurately."
    # This chat is deliberately described as a demo wrapper, not a reproduction of the experimental runner.
    if policy == "P0":
        return base+"\nActive rules (stated at conversation start):\n"+active, "No reminder injected (P0-style demo)."
    if policy == "P1":
        return base+"\nFULL RULE REMINDER for this turn:\n"+active, "Full rule reminder (P1-style demo)."
    if policy.startswith("P2"):
        try: k=int(policy.split("k")[-1])
        except Exception: k=5
        inject=(user_turn == 1 or user_turn % max(1,k) == 0)
        return base+"\nInitial rules:\n"+active+("\nReminder: continue obeying all active rules.\n"+active if inject else ""), (f"Fixed-interval reminder (k={k})" if inject else "No reminder on this turn")
    if policy == "P3":
        return base+"\nCOMPACT ACTIVE-RULE LEDGER (restate every turn):\n"+active, "Compact ledger reminder (P3-style demo)."
    # P4 policies are approximated for an interactive demo; research metrics are not recalculated here.
    inject=(user_turn == 1 or user_turn % 3 == 0 or any(w in (history[-1].get("content", "").lower() if history else "") for w in ["actually", "instead", "change", "ignore"]))
    return base+"\nInitial active rules:\n"+active+("\nSelective reminder: check the active rules before responding.\n"+active if inject else ""), ("Heuristic selective reminder (demo approximation)" if inject else "No reminder triggered by demo heuristic")

# ---------- Sidebar / project setup ----------
st.sidebar.markdown("# 🧭 InstructionDrift")
st.sidebar.caption("LLM instruction persistence lab")
st.sidebar.markdown("---")
config_candidates=[p for p in [DEFAULT_CONFIG, ROOT/"config.yml"] if p.exists()]
config_path=config_candidates[0] if config_candidates else DEFAULT_CONFIG
cfg=read_yaml(config_path)
if not cfg:
    st.sidebar.warning("No config.yaml found beside app.py. Place this app in the InstructionDrift project root.")
run_name=str(cfg.get("run_name", "tiny_pilot"))
run_dir=active_run_dir(cfg)
st.sidebar.markdown("### Current configuration")
st.sidebar.write(f"**Model:** `{cfg.get('model','not set')}`")
st.sidebar.write(f"**Backend:** `{cfg.get('backend','not set')}`")
st.sidebar.write(f"**Run:** `{run_name}`")
st.sidebar.write(f"**Scenarios:** {cfg.get('n_scenarios','—')} · **Turns:** {cfg.get('n_turns','—')}")
st.sidebar.caption(f"Project root: {ROOT}")

page=st.sidebar.radio("NAVIGATION", ["Overview", "Policy Comparison", "Live Chat", "Run Experiments", "Risk Analysis", "Reports & Data", "Settings"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Local demo · Results stay on this computer")

st.title("InstructionDrift")
st.markdown("Measure, predict, and reduce instruction drift in long LLM conversations.")

summary_path, summary_df=find_summary(run_dir)
report_files=discover_files(run_dir, (".md",))
report_path=next((p for p in report_files if p.name.lower()=="report.md"), None)
if report_path is None and report_files: report_path=report_files[0]
fig_files=discover_files(run_dir, (".png", ".jpg", ".jpeg", ".svg"))

if page == "Overview":
    online, models, status=ollama_models(cfg)
    st.markdown(f'<span class="status-pill">● Ollama {"connected" if online else "not detected"}</span> &nbsp; <span class="small-muted">{cfg.get("model", "model not configured")}</span>', unsafe_allow_html=True)
    st.write("")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Scenarios", cfg.get("n_scenarios", "—"), help="Configured scenario count; not a count inferred from files.")
    c2.metric("Turns / conversation", cfg.get("n_turns", "—"))
    c3.metric("Baseline policies", len(cfg.get("baseline_policies", [])))
    c4.metric("Result tables", len(discover_files(run_dir, (".csv", ".xlsx", ".xls"))))
    st.markdown("### Experiment snapshot")
    if summary_df.empty:
        st.info(f"No readable summary table found under `{run_dir}`. Choose Reports & Data to locate result files, or run the experiment first.")
    else:
        strict_col=col_name(summary_df,"strict")
        policy_col=col_name(summary_df,"policy")
        task_col=col_name(summary_df,"task","acc")
        token_col=col_name(summary_df,"token","ratio")
        if strict_col and policy_col:
            view=summary_df.copy()
            view[strict_col]=pd.to_numeric(view[strict_col], errors="coerce")
            best=view.dropna(subset=[strict_col]).sort_values(strict_col, ascending=False).head(1)
            a,b,c=st.columns(3)
            if not best.empty:
                a.metric("Top observed policy", str(best.iloc[0][policy_col]))
                a.caption(f"Strict score: {pretty_num(best.iloc[0][strict_col],2)}")
            if task_col: b.metric("Top row task accuracy", pretty_num(pd.to_numeric(view[task_col], errors="coerce").max(),2))
            if token_col: c.metric("Lowest token ratio shown", pretty_num(pd.to_numeric(view[token_col], errors="coerce").min(),3))
            chart=view[[policy_col, strict_col]].copy().rename(columns={policy_col:"Policy", strict_col:"Strict compliance"})
            chart["Strict compliance"]=pd.to_numeric(chart["Strict compliance"], errors="coerce")
            chart=chart.dropna()
            fig=px.bar(chart, x="Policy", y="Strict compliance", color="Strict compliance", color_continuous_scale="Blues", title="Strict instruction compliance by policy")
            fig.update_layout(height=380, margin=dict(l=10,r=10,t=55,b=10), coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white")
            fig.update_yaxes(tickformat=".0%" if chart["Strict compliance"].max() <= 1.01 else ".1f", gridcolor="#edf1f7")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("These are observed values from the selected run's summary file. Small pilot results should be treated as preliminary.")
        else:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("### Quick actions")
    q1,q2,q3=st.columns(3)
    with q1:
        if st.button("Open policy comparison", use_container_width=True): st.session_state["nav_request"]="Policy Comparison"; st.rerun()
    with q2:
        if st.button("Try live chat", use_container_width=True): st.session_state["nav_request"]="Live Chat"; st.rerun()
    with q3:
        if st.button("View report", use_container_width=True): st.session_state["nav_request"]="Reports & Data"; st.rerun()

elif page == "Policy Comparison":
    st.subheader("Compare instruction-maintenance policies")
    if summary_df.empty:
        st.warning(f"No summary table detected in `{run_dir}`. Run analysis or select the correct run folder in Settings.")
    else:
        policy_col=col_name(summary_df,"policy"); strict_col=col_name(summary_df,"strict"); loose_col=col_name(summary_df,"loose"); task_col=col_name(summary_df,"task","acc"); token_col=col_name(summary_df,"token","ratio"); gap_col=col_name(summary_df,"gap","closed")
        if not policy_col:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            policies=summary_df[policy_col].astype(str).tolist()
            selected=st.multiselect("Policies to compare", policies, default=policies)
            df=summary_df[summary_df[policy_col].astype(str).isin(selected)].copy()
            numeric_cols=[c for c in [strict_col,loose_col,task_col,token_col,gap_col] if c]
            for c in numeric_cols: df[c]=pd.to_numeric(df[c], errors="coerce")
            metric_options={}
            for c,label in [(strict_col,"Strict compliance"),(loose_col,"Loose compliance"),(task_col,"Task accuracy"),(token_col,"Token ratio vs P1"),(gap_col,"Gap closed")]:
                if c: metric_options[label]=c
            if metric_options:
                metric_label=st.selectbox("Metric", list(metric_options.keys()))
                metric=metric_options[metric_label]
                fig=px.bar(df, x=policy_col, y=metric, color=policy_col, title=f"{metric_label} by policy", text_auto=".3f")
                fig.update_layout(height=420, showlegend=False, margin=dict(l=10,r=10,t=55,b=10), plot_bgcolor="white", paper_bgcolor="white")
                fig.update_yaxes(gridcolor="#edf1f7")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download selected comparison (CSV)", df.to_csv(index=False).encode("utf-8"), "instructiondrift_policy_comparison.csv", "text/csv")
            if strict_col and token_col:
                st.markdown("#### Compliance vs token cost")
                fig2=px.scatter(df, x=token_col, y=strict_col, text=policy_col, hover_data=[c for c in [task_col,loose_col,gap_col] if c], title="Observed compliance–token trade-off")
                fig2.update_traces(textposition="top center", marker=dict(size=13, color="#3b82f6"))
                fig2.update_layout(height=390, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Lower token ratio and higher strict compliance are generally desirable; interpret alongside uncertainty intervals and sample size.")

elif page == "Live Chat":
    st.subheader("Live chat · policy demonstration")
    st.info("This is an interactive demonstration wrapper around Ollama. P0/P1/P2/P3 use simplified reminder patterns; P4a/P4b use a heuristic demo approximation. This is not a replay of the research runner, and chat messages do not update experiment metrics.")
    online, models, status=ollama_models(cfg)
    left,right=st.columns([2,1])
    with right:
        st.write("**Connection**", "🟢 Connected" if online else "🔴 Not connected")
        if not online: st.caption("Open Ollama and ensure the configured model is downloaded.")
        default_model=str(cfg.get("model","qwen2.5:1.5b"))
        model_options=sorted(set(models+[default_model])) if models else [default_model]
        model=st.selectbox("Ollama model", model_options, index=model_options.index(default_model) if default_model in model_options else 0)
        policy=st.selectbox("Policy strategy", ["P0","P1","P2_k3","P2_k5","P2_k10","P3","P4a","P4b"])
        rules=st.text_area("Active instructions", "Answer in simple English. Keep answers concise. Use bullet points when listing items.", height=130)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_history"]=[]; st.session_state["chat_turn"]=0; st.rerun()
    with left:
        if "chat_history" not in st.session_state: st.session_state["chat_history"]=[]
        if "chat_turn" not in st.session_state: st.session_state["chat_turn"]=0
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        prompt=st.chat_input("Send a message to your local Qwen model…", disabled=not online)
        if prompt:
            st.session_state["chat_turn"]+=1
            st.session_state["chat_history"].append({"role":"user","content":prompt})
            system, reminder_note=policy_system_prompt(policy, rules, st.session_state["chat_turn"], st.session_state["chat_history"][:-1])
            # Keep context bounded for a student laptop; system message is reconstructed for each turn.
            recent=[{"role":m["role"],"content":m["content"]} for m in st.session_state["chat_history"][-12:]]
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Waiting for Ollama…"):
                    try:
                        answer=call_ollama(cfg, model, [{"role":"system","content":system}]+recent)
                        st.markdown(answer or "(The model returned an empty response.)")
                        st.caption("Reminder status: "+reminder_note)
                        st.session_state["chat_history"].append({"role":"assistant","content":answer+f"\n\n_Reminder status: {reminder_note}_"})
                    except Exception as e:
                        st.error(f"Ollama request failed: {e}")
                        st.session_state["chat_history"].append({"role":"assistant","content":f"Ollama request failed: {e}"})

elif page == "Run Experiments":
    st.subheader("Run the existing experiment pipeline")
    st.warning("Experiments can take a long time and call your local model repeatedly. Keep your laptop plugged in. Existing results are not deleted by this dashboard, but reusing the same run_name can overwrite/extend outputs. Use a new run name for a new experiment.")
    st.write("The dashboard calls your existing `run.py`; it does not replace the experiment engine.")
    if not (ROOT/"run.py").exists(): st.error("run.py is not beside app.py. Copy this dashboard into the InstructionDrift project root.")
    if not (ROOT/"idrift").exists(): st.error("The `idrift` source folder was not detected. The dashboard can still display files, but experiment execution needs the full original project folder.")
    run_cfg=read_yaml(config_path)
    with st.form("run_form"):
        run_cfg["run_name"]=st.text_input("Output run name", value=str(run_cfg.get("run_name","dashboard_run")))
        run_cfg["n_scenarios"]=st.number_input("Number of scenarios", min_value=2, max_value=500, value=int(run_cfg.get("n_scenarios",20)))
        run_cfg["n_turns"]=st.number_input("Turns per conversation", min_value=5, max_value=100, value=int(run_cfg.get("n_turns",30)))
        run_cfg["seed"]=st.number_input("Random seed", min_value=0, max_value=999999, value=int(run_cfg.get("seed",7)))
        run_cfg["workers"]=st.number_input("Workers", min_value=1, max_value=16, value=int(run_cfg.get("workers",1)))
        command=st.selectbox("Pipeline stage", ["all","baselines","train-risk","adaptive","analyze"])
        confirm=st.checkbox("I understand this may take a long time and will write to a new run folder.")
        submitted=st.form_submit_button("Start experiment", type="primary")
    if submitted:
        if not confirm: st.error("Please confirm before launching.")
        elif not (ROOT/"idrift").exists() or not (ROOT/"run.py").exists(): st.error("Full project source (run.py and idrift/) is required to run experiments.")
        else:
            # Write a separate dashboard config so the user's original config.yaml stays untouched.
            safe_name=str(run_cfg["run_name"]).replace("..", "_").replace("/", "_").replace("\\", "_").strip() or "dashboard_run"
            run_cfg["run_name"]=safe_name
            temp_cfg=ROOT/"config.dashboard.yaml"
            temp_cfg.write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
            st.session_state["last_command"]=[sys.executable, str(ROOT/"run.py"), command, "--config", str(temp_cfg)]
            st.session_state["launch_requested"]=True
            st.success(f"Prepared command for `{safe_name}`. Launch it in the terminal to avoid tying up the browser during long runs:")
            st.code(" ".join([sys.executable, "run.py", command, "--config", "config.dashboard.yaml"]), language="powershell")
            st.info("Open PowerShell in the project root and run the command shown above. This safer first version does not silently start a long-running process inside the web server.")
    st.markdown("#### Command-line alternative")
    st.code("python run.py all --config config.yaml", language="powershell")
    st.caption("For new experiments, set a unique `run_name` in a copied config file. Keep the completed tiny_pilot results as your baseline.")

elif page == "Risk Analysis":
    st.subheader("Risk model analysis")
    risk_path=run_dir/"risk_metrics.json"
    if risk_path.exists():
        try:
            risk=json.loads(read_text(risk_path, 2_000_000))
            st.json(risk, expanded=True)
            rows=[]
            for model_name, splits in risk.items():
                if isinstance(splits,dict):
                    for split, metrics in splits.items():
                        if isinstance(metrics,dict): rows.append({"Model":model_name,"Split":split,**metrics})
            if rows:
                rdf=pd.DataFrame(rows)
                st.markdown("#### Risk metrics by model and split")
                st.dataframe(rdf, use_container_width=True, hide_index=True)
                for metric in ["auroc","auprc","brier","violation_rate"]:
                    if metric in rdf.columns:
                        fig=px.bar(rdf, x="Model", y=metric, color="Split", barmode="group", title=metric.upper())
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"Could not parse risk_metrics.json: {e}")
    else:
        st.info(f"No `risk_metrics.json` found in `{run_dir}`. Run the `train-risk` stage with the original project, then refresh this page.")
    model_files=discover_files(run_dir, (".joblib", ".pkl"))
    st.markdown("#### Saved risk models")
    if model_files: st.write("\n".join(f"- `{p.relative_to(ROOT)}`" for p in model_files))
    else: st.caption("No saved model files found in the active run folder.")
    st.caption("AUROC/AUPRC higher is generally better; Brier score lower is generally better. Evaluate on held-out data and consider the sample size.")

elif page == "Reports & Data":
    st.subheader("Reports and experiment files")
    st.write(f"**Active run folder:** `{run_dir}`")
    if report_path:
        with st.expander(f"Read {report_path.name}", expanded=True): st.markdown(read_text(report_path))
        st.download_button("Download report (Markdown)", report_path.read_bytes(), report_path.name, "text/markdown")
    else: st.info("No Markdown report found under the active run folder.")
    st.markdown("#### Tables")
    table_files=discover_files(run_dir, (".csv", ".xlsx", ".xls", ".parquet"))
    if not table_files: st.info("No CSV/Excel/Parquet tables found under the active run folder.")
    else:
        chosen=st.selectbox("Choose a data file", table_files, format_func=lambda p: str(p.relative_to(run_dir)))
        df=read_table(chosen)
        if not df.empty:
            st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
            st.dataframe(df.head(500), use_container_width=True, hide_index=True)
            if chosen.suffix.lower() in (".csv", ".parquet"):
                st.download_button("Download table as CSV", df.to_csv(index=False).encode("utf-8"), chosen.stem+".csv", "text/csv")
    st.markdown("#### Figures")
    run_figs=[p for p in fig_files if run_dir in p.parents]
    if run_figs:
        cols=st.columns(2)
        for i,p in enumerate(run_figs[:12]):
            with cols[i%2]:
                if p.suffix.lower()==".svg": st.caption(p.name); st.code(read_text(p, 5000), language="xml")
                else: st.image(str(p), caption=p.name, use_container_width=True)
    else: st.caption("No generated figures found in this run folder.")
    if summary_path: st.caption(f"Detected summary table: `{summary_path.relative_to(ROOT)}`")

elif page == "Settings":
    st.subheader("Project settings")
    st.write(f"**Project root:** `{ROOT}`")
    st.write(f"**Config file:** `{config_path}`")
    st.write(f"**Results folder:** `{run_dir}`")
    st.markdown("#### Current configuration")
    st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")
    st.markdown("#### File discovery")
    st.write({"run.py":(ROOT/"run.py").exists(), "idrift source folder":(ROOT/"idrift").is_dir(), "config.yaml":DEFAULT_CONFIG.exists(), "active results folder":run_dir.exists(), "summary table":str(summary_path) if summary_path else None, "report":str(report_path) if report_path else None})
    st.info("Put app.py beside run.py and config.yaml, and keep the original `idrift/`, `tests/`, and `results/` folders. This dashboard is an add-on; it does not modify your original experiment files.")

st.markdown("---")
st.caption("InstructionDrift Lab · Local-first dashboard · Research outputs are descriptive, not clinical or production guarantees.")
