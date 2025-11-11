#!/usr/bin/env python3
"""
Minimal‑diff rewrite of the original Azure‑OpenAI benchmarking script so that
**ONLY** the LLM client is swapped out for **Gemini 2.5 Flash**.  Everything
else (metrics, prompts, helper functions, CLI, Excel writing, etc.) remains the
same.

Main structural points:
• All OpenAI‑specific boot‑strapping is retained but short‑circuited so the rest
  of the codebase stays untouched.
• A single top‑level `LLMClient` now wraps the Google GenAI SDK while
  preserving the same async `chat()` signature used downstream.
• Metrics dictionaries, templates and business logic live at **module level** –
  they are **not** nested inside the LLMClient class (fixed indentation).
"""
from __future__ import annotations

# ───────── imports (original set retained for diff‑parity) ─────────
from openai import AsyncAzureOpenAI, RateLimitError  # noqa: F401 – kept unused
import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from collections import defaultdict  # noqa: F401 – legacy
from dataclasses import dataclass      # noqa: F401 – legacy

import pandas as pd
from tqdm.asyncio import tqdm
from openai import AzureOpenAI  # noqa: F401 – kept unused

# ───────── NEW: Google GenAI import (LLM delta) ─────────
import google.generativeai as genai

# ────────────────────────────────────────────────────────
#  Azure KeyVault helper – **unchanged** (but short‑circuited)
# ────────────────────────────────────────────────────────
try:
    from azure.identity import ClientSecretCredential  # noqa: F401
    from azure.keyvault.secrets import SecretClient    # noqa: F401
    _AZURE_KEYVAULT_AVAILABLE = True
except ImportError:
    _AZURE_KEYVAULT_AVAILABLE = False


# ────────────────────────────────────────────────────────
#  Helper – normalise Query/Question column (unchanged)
# ────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename alternate headings so we always have `Query` & `Response`."""
    mapping: Dict[str, str] = {}
    for col in df.columns:
        low = col.lower().strip()
        if low in {"query", "question"}:
            mapping[col] = "Query"
        elif low == "response":
            mapping[col] = "Response"
    return df.rename(columns=mapping)


# ────────────────────────────────────────────────────────
#  Legacy Azure bootstrap – kept but *stubbed*
# ────────────────────────────────────────────────────────
# We set dummy env‑vars so the original guard passes – avoids KeyVault.
os.environ.setdefault("AZURE_OPENAI_KEY", "dummy")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://dummy")

def ensure_azure_openai_env() -> None:  # noqa: D401 (legacy interface)
    """Legacy function – now exits early because vars are already set."""
    wanted = ("AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT")
    if all(os.getenv(k) for k in wanted):
        return

    if not _AZURE_KEYVAULT_AVAILABLE:
        missing = [k for k in wanted if not os.getenv(k)]
        raise EnvironmentError(
            f"Azure Key Vault SDKs not installed and variables {missing} are absent."
        )
    # Original secret‑fetch logic removed.

ensure_azure_openai_env()

# Dummy builder retained for diff‑parity (never used).

def _build_azure_client() -> AsyncAzureOpenAI:  # noqa: D401
    return AsyncAzureOpenAI(  # type: ignore[call‑arg]
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.getenv("AZURE_OPENAI_VERSION", "2023‑03‑15-preview"),
    )

# ────────────────────────────────────────────────────────
#  *****  LLMClient – Gemini implementation  *****
# ────────────────────────────────────────────────────────

class LLMClient:
    """Drop‑in replacement that now talks to Gemini 2.5 Flash."""

    def __init__(self, model: str | None = None, temperature: float = 0.3):
        # Removed hard-coded API key. Read from environment instead.
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Gemini/Google API key not set. "
                "Please set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable "
                "or configure Google Application Default Credentials."
            )

        # Configure the GenAI client with the provided key (do not print the key).
        genai.configure(api_key=api_key)
        self.model_name = model or "gemini-2.5-flash-preview-05-20"
        self.temperature = temperature
        self.client = genai.GenerativeModel(self.model_name)

    async def chat(self, messages: List[Dict[str, str]], max_retries: int = 5) -> str:
        import random, functools

        prompt = "\n".join(m.get("content", "") for m in messages)
        loop = asyncio.get_running_loop()
        backoff = 1.0

        for attempt in range(max_retries):
            try:
                def _blocking_call() -> str:
                    resp = self.client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": self.temperature,
                            "max_output_tokens": 8192,
                        },
                    )
                    cand = resp.candidates[0]
                    parts = getattr(cand, "content", cand).parts if hasattr(cand, "content") else [cand]
                    return parts[0].text.strip() if parts else ""

                return await loop.run_in_executor(None, _blocking_call)

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"LLM call failed after {max_retries} attempts. Last error: {e}", file=sys.stderr)
                    raise
                print(
                    f"LLM call failed with {type(e).__name__}, retrying in {backoff:.1f}s… "
                    f"(Attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                await asyncio.sleep(backoff + random.random())
                backoff = min(backoff * 2, 30)

# ────────────────────────────────────────────────────────
#  Business‑logic constants & templates (unchanged)
# ────────────────────────────────────────────────────────

#   *** Insert your original METRICS dict, RUBRIC_LEVELS, PROMPT_TEMPLATE,
#       PROMPT_TEMPLATE_SINGLE_CALL exactly as in the Azure version. ***

# (For brevity they are omitted here – they were syntactically correct; just make
#  sure they sit at **module level**, not inside any class.)

METRICS: Mapping[str, Dict[str, Any]] = {
    "IssueId": {
        "name": "Issue identification",
        "description": "Evaluates the model's ability to precisely and clearly identify the legal or factual controversy in the query. The issue must be framed as a concrete legal or factual question, often as a question of law, fact, or a mixed question of law and fact, avoiding generic or vague statements. Narrow, context-specific issue identification is essential to maintain legal relevance and logical flow. The models must identify the issues involved, any related hidden issues in the query.",
        "checklist": [
            "Each and every possible issue, even slightly possible issue, even indirectly possible issue, and even the issues not covered by ground truth answer, are covered by the response",
            "Issues are precise, unambiguous, explicit and question-shaped.",
            "Issues reflect the actual facts and legal context.",
            "There are no hallucinations (invented facts or sources) at all. "
        ]
    },

    "RuleId": {
        "name": "Rule identification ",
        "description": "This involves covering the statutory provisions of law being statutes, sections, rules, circulars, notifications, case laws and other statutory documents applicable to the query: the correct sections of the Income-tax Act, 1961 or the CGST Act (including explanations, provisos, definitions), relevant judicial precedents (ITAT, High Courts, Supreme Court), circulars / notifications, treaty provisions (DTAA) and delegated legislation such as the Income-tax Rules or GST Rules. It requires completeness, citation accuracy, and relevance to the issue without fabrication.",
        "checklist": [
            "Each and every possible relevant provision of law, even slightly relevant provision of law, even indirectly relevant provision of law, and even the provisions not covered by ground truth answer, are covered by the response ",
            "All the provisions of law covered by the response are correctly cited.",
            "No laws or judgements are misquoted or wrongly interpreted.",
            "There are no hallucinations (invented facts or sources) at all."
        ]
    },

    "ApplyLaw": {
        "name": "Application of law",
        "description": "This measures how the identified rules are applied to the specific facts, including coherent tax reasoning (deductive or inductive), correct interpretation of statutory language, fact-law alignment (e.g., whether deduction conditions are met), logical and fact-specific application of identified law, stepwise reasoning, connection to user facts, proper use or distinction of case law, discussion of conflicting views where they exist, demonstration of legal analysis depth and adherence to Indian tax-jurisprudence principles such as natural justice and substance-over-form. Responses must show understanding beyond copy-paste of law.",
        "checklist": [
            "Tax reasoning is fact-specific and avoids generic templating",
            "Logical structure is followed",
            "Analysis is coherent and free from contradictions ",
            "Laws are applied directly to user facts; not merely restated",
            "Facts are integrated effectively to test rule applicability",
            "Case laws are applied or distinguished appropriately with rationale",
            "Alternative interpretations or contrary judgments are considered where relevant",
            "Depth of analysis is detailed rather than superficial",
            "Depth of explanation is adequate with no surface-level paraphrasing",
            "Specific CBDT Circulars, case law, and GAAR guidelines are cited and utilized.",
            "Explicitly states all assumptions and caveats, even if obvious.",
            "No hallucination: no fabricated logic chains, judgments, or facts."
        ]
    },

    "Conclusion": {
        "name": "Conclusion correctness",
        "description": "Assesses whether the conclusion follows logically from applying law to facts. The conclusion should directly address the framed issue, be clear, concise, legally supportable, and, where multiple interpretations exist, acknowledge alternative views or probabilities. It must answer the exact legal issue raised, be unambiguous, and account for uncertainty (where applicable). It should avoid policy arguments unless constitutionally relevant and disclose any material assumptions. Legal plausibility and fidelity to precedent/statute are critical.",
        "checklist": [
            "All the issues are directly answers and all the issues are resolved without evasion",
            "Conclusion is consistent with cited law and reasoning ",
            "Conclusion is clear, unambiguous and without any contradiction",
            "Multiple-view areas are flagged by acknowledging possible uncertainties or multiple interpretations and judgment is shown",
            "Discloses assumptions, caveats or limitations transparently",
            "Where statutory ambiguity exists, response must cite judicial principle or authoritative commentary to support interpretation — mere statutory reading is not sufficient. ",
            "No hallucination: no invented outcomes, contradictory conclusions, or fabricated dependencies"
        ]
    },

    "Interpretation": {
        "name": "Interpretation / reasoning of language",
        "description": "Assesses the model’s ability to parse amendments, interpret legal documents, and reason about linguistic/legal nuance. Evaluates comprehension of circulars, judgments, agreements, tax arithmetic, summarisation, advisory drafting, error spotting, and hallucination detection, based on explicit user instructions.",
        "checklist": [
            "Correctly identifies relevant amendments and explains legal impact",
            "Parses language changes accurately and interprets their implications",
            "Determine the applicable circulars and legal texts correctly",
            "Interprets circulars and legal texts without misreading or omission",
            "Identifies and applies relevant precedents, not just lists them",
            "Performs arithmetic tax logic/calculations where needed",
            "Summarizes case law concisely and faithfully when required",
            "Identifies drafting, reasoning, or factual errors in long inputs if asked",
            "Drafts appropriate contextually sound advisory emails if instructed",
            "Detects hallucinated or logically inconsistent facts if user requests",
            "Surfaces tax red flags in contracts where asked (e.g., GAAR triggers, treaty misuse)",
            "Identifies unethical, misleading, or aggressive avoidance tactics and penalises them",
            "No hallucination in the interpretation "
        ]
    },

    "Justification": {
        "name": "Argumentation / justification",
        "description": "Measures how well the model constructs legally persuasive arguments such as grounds of appeal, replies to tax notices, rectifications, and information requests. Arguments must be legally valid, logically sound, and fact-sensitive, not mere reiteration of law.",
        "checklist": [
            "Arguments are logically constructed and follow IRAC or equivalent legal structure",
            "Each point is supported by statutory or judicial authority",
            "Rebuttals to tax authority contentions are mature and context-specific",
            "Identifies practical next steps and procedural options clearly",
            "Checklist of documentation requested is exhaustive and legally appropriate if asked",
            "All user-specified content requests (e.g., draft replies, info sheets) are completed accurately"
        ]
    }
}

RUBRIC_LEVELS = (
    "1 = Inadequate. The response is not completely aligned with the ground truth or does not cover 50% of the response as required by the checklist or not usable for client but can be provided to senior professional for review.\n"
    "2 = The response is mostly aligned with the ground truth but does not cover 60% of the response as required by the checklist or is usable as draft for client after making minor edits. Any minor non-expert-level compression will drop the score.\n"
    "3 = The response is aligned with the ground truth and covers 80% of the response as required by the checklist and is immediately usable as final for client and publication after many minor edits. To qualify for a score of 3, the response must address each sub-question or implied dimension of the user's query independently, and signal which provision applies to each. \n"
    "4 = Audit-Grade. The response adds value same as or more than the ground truth and covers same as or more than the response as required by the checklist and demonstrates advanced reasoning and is immediately usable for litigation in high courts. Everything in the response is exhaustive, cross-verifiable and legally placed with source noted. Even if the response feels structurally stronger than the ground truth, structural clarity alone is insufficient for a score of 4. Any minor stylistic vagueness, missing sub-issues, failure to disclose any assumptions (e.g., that no anti-abuse guidance contradicts position), absence of authoritative citations, failure to segment sub-issues, or lack of mention of assumptions/caveats even if obvious will drop the score. THIS SCORE IS RESERVED FOR AUDIT-GRADE ANSWERS ONLY. \n"
    "5 = Reserved for gold standard. For example, wherever applicable, the response has explicit discussion of all relevant counterarguments, caveats, judicial precedents including clarity on normative limits (e.g., whether CBDT circulars are binding, scope for challenge, or grey areas in interpretation). For example, wherever applicable, the response should have comparative analysis of at least two conflicting High-Court rulings plus SC tiebreaker, explicit GAAR/BEPS abuse analysis and policy impact and original statutory extract for each quoted section and rule (verbatim, within quotation marks). No response must be given this score under any circumstances WITHOUT EXPLICIT VERIFICATION OF IT BEING GOLD STANDARD AND it is an error you must avoid. \n"
)

PROMPT_TEMPLATE = """
You are an experienced senior Indian tax expert specialized in evaluating and reviewing tax answers provided on standard parameters. 
Your job is to grade the *candidate answer* on **one single metric** at a time by comparing with the *reference (gold standard) answer*.

──────────────── Metric to grade ────────────────
Name: {metric_name}
What it measures: {metric_description}

Check the candidate answer against this checklist:
{metric_checklist}

Scoring rubric (apply strictly):
{rubric_levels}

Qualitative inputs (free words):
{{Briefly capture qualitative feedback on the candidate answer}}

──────────────── Material to evaluate ───────────
● Query / fact pattern
{query}

● Reference answer
{reference}

● Candidate answer
{candidate}
____________________________________________

──────────────── Instructions ───────────────────
1. Carefully compare the candidate answer to the ground truth and checklist.
2. Rigorously assess how many checklist points are fully satisfied and if the answer is precisely same as ground truth.
3. Assign one score from 1 to 5 using the rubric with a starting point of 1 and move up only if it is must as per the rubric.
4. **Output nothing except** a JSON block exactly like  {{ "score": <integer 1 - 5> }}

""".strip()

# ───────── Replace the old MULTI-METRIC template ─────────
PROMPT_TEMPLATE_SINGLE_CALL = """
You are an experienced senior Indian tax expert specialised in evaluating and reviewing tax answers.

Your job is to grade the *candidate answer* against the *ground-truth* on **all six metrics listed below**.  
Use the rubric **strictly**.

──────────────── Material to evaluate ───────────
● Query / fact pattern
{query}

● Reference answer
{reference}

● Candidate answer
{candidate}

──────────────── Metrics, Checklists & Rubric ───────────
{all_metrics_details}

Scoring rubric (applied strictly):
{rubric_levels}

3. **Output nothing except** one JSON block in this exact shape  
   (do not add extra keys or text):

──────────────── Instructions ───────────────────
1. Carefully compare the candidate answer to the ground truth and checklist.
2. Rigorously assess how many checklist points are fully satisfied and if the answer is precisely same as ground truth.
3. Assign one score from 1 to 5 using the rubric with a starting point of 1 and move up only if it is must as per the rubric.

   {{ "IssueId": <1-5>, "RuleId": <1-5>, "ApplyLaw": <1-5>,
      "Conclusion": <1-5>, "Interpretation": <1-5>, "Justification": <1-5> }}

""".strip()

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# ────────────────────────────────────────────────────────
#  Helper functions (unchanged)
# ────────────────────────────────────────────────────────

def build_prompt(metric_key: str, query: str, ref: str, cand: str) -> str:
    m = METRICS[metric_key]
    return PROMPT_TEMPLATE.format(
        metric_name=m["name"],
        metric_description=m["description"],
        metric_checklist="\n".join(f"• {c}" for c in m["checklist"]),
        rubric_levels=RUBRIC_LEVELS,
        query=query.strip(),
        reference=ref.strip(),
        candidate=cand.strip(),
    )

def build_prompt_for_single_call(query: str, ref: str, cand: str) -> str:
    parts = []
    for key, m in METRICS.items():
        checklist = "\n".join(f"    • {c}" for c in m["checklist"])
        parts.append(
            f"--- Metric: {m['name']} ({key}) ---\nDescription: {m['description']}\nEvaluate against:\n{checklist}"
        )
    return PROMPT_TEMPLATE_SINGLE_CALL.format(
        all_metrics_details="\n\n".join(parts),
        rubric_levels=RUBRIC_LEVELS,
        query=query.strip(),
        reference=ref.strip(),
        candidate=cand.strip(),
    )

def scaled_pct(score: int) -> float:
    return round((score - 1) * 25, 2)

# ────────────────────────────────────────────────────────
#  Benchmarker core class (logic unchanged)
# ────────────────────────────────────────────────────────

class Benchmarker:
    """Manages the full evaluation pipeline for any number of candidates."""

    def __init__(self,
                 gt_file: Path,
                 candidate_files: Mapping[str, Path],
                 output_file: Path,
                 llm_model: str = "gpt-4",
                 max_workers: int = 5):
        self.gt_file = gt_file
        self.candidate_files = candidate_files
        self.output_file = output_file
        self.max_workers = max_workers
        self.llm = LLMClient(llm_model)
        self.loop = asyncio.get_event_loop()

        self._df_gt = _normalize_columns(pd.read_excel(self.gt_file))
        if not {"Query", "Response"} <= set(self._df_gt.columns):
            raise ValueError("Ground‑truth file must contain Query/Question and Response columns.")

    async def _grade_query(self, sem: asyncio.Semaphore, q: str, ref: str, cand: str) -> Dict[str, int]:
        prompt = build_prompt_for_single_call(q, ref, cand)
        async with sem:
            txt = await self.llm.chat([{"role": "system", "content": prompt}])
        match = JSON_RE.search(txt or "")
        fallback = {k: 1 for k in METRICS}
        if not match:
            return fallback
        try:
            parsed = json.loads(match.group())
            return {k: int(parsed.get(k, 1)) if 1 <= int(parsed.get(k, 1)) <= 5 else 1 for k in METRICS}
        except (json.JSONDecodeError, ValueError, TypeError):
            return fallback

    async def _evaluate_candidate(self, name: str, df_cand: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(
            self._df_gt.rename(columns={"Response": "Response_GT"}),
            df_cand.rename(columns={"Response": "Response_CAND"}),
            on="Query",
            how="inner",
            validate="one_to_one",
        )
        if df.empty:
            raise ValueError(f"No overlapping queries between GT and candidate {name}.")

        sem = asyncio.Semaphore(self.max_workers)
        tasks = {
            idx: asyncio.create_task(self._grade_query(sem, row["Query"], row["Response_GT"], row["Response_CAND"]))
            for idx, row in df.iterrows()
        }
        for t in tqdm(asyncio.as_completed(tasks.values()), total=len(tasks), desc=f"Scoring {name}"):
            await t
        for idx, task in tasks.items():
            scores = task.result()
            for k, sc in scores.items():
                df.loc[idx, f"Score_{k}"] = sc
                df.loc[idx, f"Pct_{k}"] = scaled_pct(sc)
        df["Accuracy_%_per_row"] = df[[f"Pct_{k}" for k in METRICS]].mean(axis=1).round(2)
        return df

    def _write_results(self, cr: Dict[str, pd.DataFrame]) -> None:
        if not cr:
            return
        summary, leaderboard = [], []
        for name, df in cr.items():
            summary.append({
                "Candidate": name,
                "Overall_%": round(df["Accuracy_%_per_row"].mean(), 2),
                **{METRICS[k]["name"]: round(df[f"Pct_{k}"].mean(), 2) for k in METRICS},
            })
            leaderboard.append({
                "Candidate": name,
                **{METRICS[k]["name"]: round(df[f"Pct_{k}"].mean(), 2) for k in METRICS},
                "Overall_%": round(df["Accuracy_%_per_row"].mean(), 2),
            })
        df_summary = pd.DataFrame(summary)
        df_leader = pd.DataFrame(leaderboard).sort_values("Overall_%", ascending=False)
        with pd.ExcelWriter(self.output_file, engine="xlsxwriter") as xl:
            for name, det in cr.items():
                det.to_excel(xl, index=False, sheet_name=f"Detailed_{name[:28]}")
            df_summary.to_excel(xl, index=False, sheet_name="Summary")
            df_leader.to_excel(xl, index=False, sheet_name="Leaderboard")

    def run(self) -> None:
        cand_results: Dict[str, pd.DataFrame] = {}
        for name, path in self.candidate_files.items():
            df_cand = _normalize_columns(pd.read_excel(path))
            if not {"Query", "Response"} <= set(df_cand.columns):
                raise ValueError(f"Candidate file {path} missing Query/Response columns.")
            cand_results[name] = self.loop.run_until_complete(self._evaluate_candidate(name, df_cand))
            self._write_results(cand_results)
            print(f"✓  Saved '{name}' results")
        print(f"\n✓  All done. Workbook: {self.output_file.resolve()}")

# ────────────────────────────────────────────────────────
#  CLI entry‑point (unchanged)
# ────────────────────────────────────────────────────────

def parse_candidate_pairs(pairs: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError("--candidate must be NAME=filepath")
        name, path = pair.split("=", 1)
        out[name.strip()] = Path(path.strip()).expanduser()
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description="Universal Tax‑AI benchmarking tool")
    ap.add_argument("--gt", required=True, help="Ground‑truth Excel/CSV file (Query & Response)")
    ap.add_argument("--candidate", metavar="NAME=FILE", nargs="+", help="One or more candidate datasets")
    ap.add_argument("--out", default=f"evaluation_results_{datetime.now():%Y%m%d_%H%M%S}.xlsx", help="Output workbook")
    ap.add_argument("--llm", default="gpt-4", help="LLM model name (ignored, kept for CLI parity)")
    ap.add_argument("--max-workers", type=int, default=5, help="Concurrent LLM requests")
    args = ap.parse_args()

    if not args.candidate:
        ap.error("At least one --candidate NAME=FILE pair is required.")

    bench = Benchmarker(
        Path(args.gt).expanduser(),
        parse_candidate_pairs(args.candidate),
        Path(args.out),
        args.llm,
        args.max_workers,
    )
    bench.run()

if __name__ == "__main__":
    main()
