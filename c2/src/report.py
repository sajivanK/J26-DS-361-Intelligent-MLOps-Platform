"""
Business report (FR7 / Objective 5): a readable model health report per run.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Pipeline
  1. report_evidence()  turns one stored run into a small set of aggregate
                        facts in business language (no data rows ever leave
                        C2, NFR5). There are no labels in production, so
                        accuracy is never measured: the report states the share
                        of the model's decisions the root cause changes, C1's
                        estimated accuracy loss when C1 is real, the affected
                        group in plain words, how many recent runs agree, and
                        the run-by-run trend.
  2. decision()         rule-based, never written by the language model:
                        no action / watch / approve targeted repair, with owners.
  3. writer             OpenAI (if OPENAI_API_KEY is set) writes four short
                        sections from the evidence only; otherwise a template.
  4. claim_check()      every number in the text must match the evidence, and
                        no feature may be named that the evidence does not
                        contain. Any failure -> template text, reason recorded.
  5. charts             data for the report visuals: decisions changed per run,
                        decisions changed by each factor, affected-group share,
                        reliance shift.

Configuration (environment variables, never committed):
    OPENAI_API_KEY   enables the language-model writer
    OPENAI_MODEL     optional, default below
    OPENAI_BASE_URL  optional: any OpenAI-compatible service, e.g. free ones
                     Gemini  https://generativelanguage.googleapis.com/v1beta/openai/
                     Groq    https://api.groq.com/openai/v1
                     Ollama  http://localhost:11434/v1   (local, key = "ollama")
    OPENAI_REASONING_EFFORT  optional thinking level for models that think before
                     answering ("low", "medium", ...). Default "low" for Gemini: on
                     the C2 report it halves the time (about 17 s instead of 32 s
                     for gemini-3.7-flash, measured 23 Oct 2026); not sent otherwise.
"""
import json
import os
import re
from datetime import datetime, timezone

from c2.src.history import c1_points, decision_points, harm_points
from c2.src.plain import NAMES, PlainLanguage

DEFAULT_MODEL = "gpt-4o-mini"
MALIGNANT_PP = 2.0
DECISION_LINE_PCT = 2.0     # 2% of decisions changed ~ the most that could reach the 2 pp harm line
SECTIONS = ["summary", "what_changed", "who_is_affected", "outlook"]
SECTION_TITLES = {"summary": "Summary", "what_changed": "What changed",
                  "who_is_affected": "Who is affected", "outlook": "What happens next"}

SYSTEM_PROMPT = """You write short model health reports for business managers who
are not data scientists. You receive JSON evidence from a drift diagnosis of a
production income-prediction model. Write four sections as a JSON object with
keys "summary", "what_changed", "who_is_affected", "outlook".

Rules:
- Use ONLY facts and numbers present in the evidence. Do not compute new
  numbers, do not estimate money, do not round differently than given.
- 1 to 3 plain sentences per section. No jargon: never say SHAP, z-score,
  EWMA, drift impact score, or feature. Say "factor" or name it in words.
- Use the plain names given (e.g. "capital gain"), not code names.
- There are no labels: accuracy is never measured. Describe the damage as the
  share of the model's decisions that change. Mention an accuracy loss ONLY if
  estimated_accuracy_loss_points is given, and call it an estimate.
- If state is "clean" or "harmless", say plainly that no repair is needed.
- If state is "suspected", say it is not confirmed yet.
- The summary MUST name the root cause in plain words, the share of decisions it
  changes and the affected group (when given), so a manager who reads only the
  summary knows what broke, how much and for whom.
- The decision is already made (evidence.decision); explain it, do not change it.
- Return only the JSON object."""


# ── 1. evidence ────────────────────────────────────────────────────────────
def report_evidence(record, history, plain=None):
    """Aggregate, plain-language facts for one run. `history` = all stored runs."""
    pl = plain or PlainLanguage()
    c3, ts, ev, loc = (record["to_c3"], record["temporal_shap"],
                       record["evidence"], record["localization"])
    state, root, n = record["state"], c3["root_cause_feature"], record.get("n_rows") or 1000
    est = harm_points(record)
    ref_now = ts["mean_abs_shap_ref"].get(root) or 0.0
    reliance = round(ts["mean_abs_shap_now"].get(root, 0.0) / ref_now, 1) if ref_now else None
    runs_agree = int(round(c3["diagnosis_stability"] * 3))
    others = sorted(((f, x) for f, x in (ev.get("decision_change_by_feature") or {}).items()
                     if f != root), key=lambda kv: kv[1], reverse=True)[:2]
    upto = [r for r in history if r["run"] <= record["run"]][-6:]
    out = {
        "run": record["run"],
        "date": record["timestamp"][:10],
        "model": "income prediction model (UCI Adult, XGBoost)",
        "state": state,
        "rows_in_batch": n,
        "accuracy_measured": False,
        "decisions_changed_percent": decision_points(record),
        "decisions_changed_rows": int(loc.get("rows_at_risk") or 0),
        "decision_line_percent": DECISION_LINE_PCT,
        "harm_line_points": MALIGNANT_PP,
        "estimated_accuracy_loss_points": None if est is None else round(est, 1),
        "trend": [{"run": r["run"], "decisions_changed_percent": decision_points(r),
                   "decisions_changed_rows": int((r.get("localization") or {}).get("rows_at_risk") or 0),
                   "estimated_accuracy_loss_points":
                   None if harm_points(r) is None else round(harm_points(r), 1),
                   "detection_step_predicted_points":
                   None if c1_points(r) is None else round(c1_points(r), 1),
                   "state": r["state"]} for r in upto],
    }
    if state != "clean":
        out.update({
            "root_cause": pl.feature(root),
            "root_cause_kind": c3["root_cause_category"].replace("_", " "),
            "model_reliance_vs_normal": reliance,
            "recent_runs_agreeing": runs_agree,
            "runs_checked": 3,
            "change_first_seen_run": record["run"] - c3["temporal_window"] + 1,
            "affected_group": pl.rule(c3["affected_slice_rule"]),
            "affected_share_percent": round(100 * c3["slice_support"], 1),
            "affected_rows": int(round(c3["slice_support"] * n)),
            "training_share_percent": (None if loc["ref_share"] is None
                                       else round(100 * loc["ref_share"], 1)),
            "next_factors": [{"factor": pl.feature(f),
                              "decisions_changed_percent": round(100 * max(x, 0), 1)}
                             for f, x in others],
        })
        pred = c1_points(record)          # real C1 only (contract v1.1: a proportion)
        if pred is not None:
            out["detection_step_predicted_points"] = round(pred, 1)
    out["decision"] = decision(out)
    return out


# ── 2. decision (rules, not the language model) ─────────────────────────────
def decision(ev):
    s = ev["state"]
    if s in ("clean", "harmless"):
        return {"level": "ok", "head": "No action needed",
                "text": "Keep the model in production and keep monitoring.",
                "actions": [{"action": "Keep the model in production", "owner": "Platform"},
                            {"action": "Check the next batch as usual", "owner": "C1, C2"}]}
    root, dec, est = ev["root_cause"], ev["decisions_changed_percent"], ev["estimated_accuracy_loss_points"]
    est_txt = f" C1 estimates about {est} points of accuracy lost." if est is not None else ""
    if s == "suspected":
        return {"level": "watch", "head": "Watch, do not repair yet",
                "text": f"A change in {root} changes about {dec}% of the model's decisions but is "
                        f"seen in one run only.{est_txt} Confirm it with the next batch before repairing.",
                "actions": [{"action": "Run the next batch to confirm the cause", "owner": "C2"},
                            {"action": "Hold any retraining until it is confirmed", "owner": "C4"},
                            {"action": f"Ask the data owner why {root} values changed",
                             "owner": "Business"}]}
    acts = [{"action": "Generate repair data for the affected group", "owner": "C3"},
            {"action": "Retrain and run the release safety check", "owner": "C4"},
            {"action": f"Ask the data owner why {root} values changed", "owner": "Business"}]
    if ev.get("next_factors"):
        acts.append({"action": f"Keep watching {ev['next_factors'][0]['factor']}, "
                               "the next most affected factor", "owner": "C2"})
    return {"level": "act", "head": "Approve targeted repair",
            "text": f"Repair the affected group, retrain, and let the release check decide. "
                    f"{root.capitalize()} changes about {dec}% of the model's decisions.{est_txt}",
            "actions": acts}


# ── 3. writers ─────────────────────────────────────────────────────────────
def template_text(ev):
    s = ev["state"]
    if s == "clean":
        return {"summary": f"The {ev['model']} handled run #{ev['run']} normally. No change in "
                           "the incoming data is changing its decisions.",
                "what_changed": "Nothing harmful. Small day-to-day changes stayed within "
                                "normal noise.",
                "who_is_affected": "No group of customers is affected.",
                "outlook": "No repair is needed. The next batch will be checked as usual."}
    root, dec, est = ev["root_cause"], ev["decisions_changed_percent"], ev["estimated_accuracy_loss_points"]
    if s == "harmless":
        detail = (f"C1 estimates only {est} points of accuracy lost, below the "
                  f"{ev['harm_line_points']} point harm line." if est is not None else
                  f"It changes only {dec}% of the model's decisions, below the "
                  f"{ev['decision_line_percent']}% line.")
        return {"summary": f"{root.capitalize()} changed in run #{ev['run']}, but it barely affects "
                           f"the model. {detail}",
                "what_changed": f"The values of {root} shifted and the model reads them "
                                "differently, but its decisions stay almost the same.",
                "who_is_affected": "No group needs repair.",
                "outlook": "No repair is needed. The change stays under watch."}
    rel = ev.get("model_reliance_vs_normal")
    summary = (f"Since run #{ev['change_first_seen_run']}, the reported {root} of incoming "
               f"customers has changed, and the model now leans on it {rel} times more than "
               f"normal. This changes about {dec}% of the model's decisions "
               f"({ev['decisions_changed_rows']} of {ev['rows_in_batch']:,} customers).")
    if est is not None:
        summary += f" C1 estimates about {est} points of accuracy lost."
    if s == "confirmed":
        what = (f"The same cause was found in {ev['recent_runs_agreeing']} of the last "
                f"{ev['runs_checked']} runs, so it is confirmed.")
    else:
        what = "This is the first run that shows the change, so it is not confirmed yet."
    if "detection_step_predicted_points" in ev:
        what += (f" The detection step also flagged the batch, predicting a "
                 f"{ev['detection_step_predicted_points']} point drop.")
    who = (f"The changed decisions concentrate in one group: {ev['affected_group']}. That is "
           f"{ev['affected_rows']} of {ev['rows_in_batch']:,} records "
           f"({ev['affected_share_percent']}%)")
    who += (f", against {ev['training_share_percent']}% of the data the model learned from."
            if ev.get("training_share_percent") is not None else ".")
    outlook = ("Repairing this group should bring most of these decisions back to normal."
               if s == "confirmed"
               else "If the next run shows the same cause, repair will be recommended.")
    return {"summary": summary, "what_changed": what, "who_is_affected": who,
            "outlook": outlook}


def writer_name():
    """openai:<model>, or <host>:<model> for another OpenAI-compatible service."""
    base = os.environ.get("OPENAI_BASE_URL", "")
    host = re.sub(r"^https?://([^/:]+).*$", r"\1", base) if base else "openai"
    return f"{host}:{os.environ.get('OPENAI_MODEL', DEFAULT_MODEL)}"


WRITER_TIMEOUT_S = 35           # one try, no automatic retry (retries use up free-tier quota)


def reasoning_effort():
    """Thinking level to request: OPENAI_REASONING_EFFORT, else "low" for Gemini, else none."""
    v = os.environ.get("OPENAI_REASONING_EFFORT")
    if v is not None:
        return v or None
    return "low" if "googleapis.com" in os.environ.get("OPENAI_BASE_URL", "") else None


def openai_text(ev, model=None, timeout=WRITER_TIMEOUT_S):
    """Four sections written by the OpenAI model. Raises on any problem."""
    from openai import OpenAI          # optional dependency, only needed here
    # a single try: a slow, busy or rate-limited service never holds the report for long;
    # after WRITER_TIMEOUT_S the template text is used instead
    client = OpenAI(timeout=timeout, max_retries=0)   # reads OPENAI_API_KEY and OPENAI_BASE_URL
    effort = reasoning_effort()
    extra = {"reasoning_effort": effort} if effort else {}
    resp = client.chat.completions.create(
        **extra,
        model=model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": json.dumps(ev)}])
    data = json.loads(resp.choices[0].message.content)
    missing = [k for k in SECTIONS if not isinstance(data.get(k), str) or not data[k].strip()]
    if missing:
        raise ValueError(f"model reply missing sections {missing}")
    return {k: data[k].strip() for k in SECTIONS}


# ── 4. claim check ─────────────────────────────────────────────────────────
_NUM = re.compile(r"(?<![\w#.])-?\d[\d,]*(?:\.\d+)?")


def _numbers(obj):
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(x.replace(",", "")) for x in _NUM.findall(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numbers(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _numbers(v)]
    return []


def claim_check(sections, ev):
    """Every number must come from the evidence; no unknown factor may be named."""
    allowed = set(_numbers(ev)) | {1000.0, 100.0}
    allowed |= {float(r) for r in range(0, 11)}            # counting words: "2 of 3"
    problems, checked = [], 0
    for key, text in sections.items():
        for x in _numbers(text):
            checked += 1
            if not any(abs(x - a) <= max(0.051, 0.005 * abs(a)) for a in allowed):
                problems.append(f"{key}: number {x:g} is not in the evidence")
    known = json.dumps(ev).lower()
    for f, name in NAMES.items():
        for key, text in sections.items():
            if re.search(rf"\b{re.escape(name)}\b", text.lower()) and name not in known:
                problems.append(f"{key}: names '{name}', which the evidence does not")
    return {"passed": not problems, "numbers_checked": checked, "problems": problems}


# ── 5. charts and assembly ─────────────────────────────────────────────────
def charts(record, plain):
    ts, ev = record["temporal_shap"], record["evidence"]
    factors = sorted(((plain.feature(f), round(100 * max(h, 0.0), 1))
                      for f, h in (ev.get("decision_change_by_feature") or {}).items()),
                     key=lambda kv: kv[1], reverse=True)[:5]
    reliance = sorted(((f, ts["mean_abs_shap_ref"][f], ts["mean_abs_shap_now"][f])
                       for f in ts["mean_abs_shap_ref"]), key=lambda t: -max(t[1], t[2]))[:8]
    return {"factors": [{"factor": f, "decisions_changed_percent": v} for f, v in factors],
            "reliance": [{"feature": f, "factor": plain.feature(f), "normal": a, "now": b}
                         for f, a, b in reliance]}


def build_report(record, history, plain=None, use_openai=True):
    pl = plain or PlainLanguage()
    ev = report_evidence(record, history, pl)
    writer, reason, sections = "template", None, None
    if use_openai and os.environ.get("OPENAI_API_KEY"):
        try:
            sections = openai_text(ev)
            check = claim_check(sections, ev)
            if check["passed"]:
                writer = writer_name()
            else:
                reason = "claim check failed: " + "; ".join(check["problems"][:3])
                sections = None
        except Exception as e:  # noqa: BLE001  network, key, quota, bad reply
            reason = f"writer {writer_name()} unavailable: {type(e).__name__}"
            sections = None
    elif use_openai:
        reason = "OPENAI_API_KEY not set"
    if sections is None:
        sections = template_text(ev)
    check = claim_check(sections, ev)
    root = ev.get("root_cause")
    title = {
        "clean": f"Run #{ev['run']}: the model is healthy",
        "harmless": f"Run #{ev['run']}: harmless change in {root}",
        "suspected": f"Run #{ev['run']}: {root} may be changing {ev['decisions_changed_percent']}% of decisions",
        "confirmed": f"Run #{ev['run']}: {root} changes {ev['decisions_changed_percent']}% of the model's decisions",
    }[ev["state"]]
    return {
        "run": ev["run"],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "writer": writer,
        "fallback_reason": reason,
        "title": title,
        "decision": ev["decision"],
        "sections": sections,
        "section_titles": SECTION_TITLES,
        "kpis": {"decisions_changed_percent": ev["decisions_changed_percent"],
                 "decisions_changed_rows": ev["decisions_changed_rows"],
                 "estimated_accuracy_loss_points": ev["estimated_accuracy_loss_points"],
                 "detection_step_predicted_points": ev.get("detection_step_predicted_points"),
                 "affected_share_percent": ev.get("affected_share_percent", 0.0),
                 "training_share_percent": ev.get("training_share_percent"),
                 "recent_runs_agreeing": ev.get("recent_runs_agreeing", 0),
                 "runs_checked": 3},
        "charts": dict(charts(record, pl), trend=ev["trend"]),
        "claim_check": check,
        "evidence": ev,
        "plain_language_decoded": pl.decoded,
    }


def report_markdown(report, record=None):
    lines = [f"# {report['title']}", "",
             f"Model health report · C2 Root Cause Diagnosis · generated {report['generated_at']}",
             f"Written by: {report['writer']}", "",
             f"**Decision: {report['decision']['head']}.** {report['decision']['text']}", ""]
    k = report["kpis"]
    est = "not available" if k["estimated_accuracy_loss_points"] is None else \
        f"{k['estimated_accuracy_loss_points']} pp (C1 estimate)"
    lines += ["| Decisions changed | Estimated accuracy loss | Customers in the group | Runs agreeing |",
              "|---|---|---|---|",
              f"| {k['decisions_changed_percent']}% | {est} | "
              f"{k['affected_share_percent']}% | {k['recent_runs_agreeing']} of {k['runs_checked']} |", "",
              "No labels exist in production, so accuracy is not measured.", ""]
    for s in SECTIONS:
        lines += [f"## {SECTION_TITLES[s]}", report["sections"][s], ""]
    lines += ["## Decisions changed per run", "", "| Run | Decisions changed (%) | Result |", "|---|---|---|"]
    lines += [f"| #{t['run']} | {t['decisions_changed_percent']} | {t['state']} |"
              for t in report["charts"]["trend"]]
    lines += ["", "## Recommended actions"]
    lines += [f"{i}. {a['action']} (owner: {a['owner']})"
              for i, a in enumerate(report["decision"]["actions"], 1)]
    c = report["claim_check"]
    lines += ["", f"Claim check: {'passed' if c['passed'] else 'FAILED'} "
                  f"({c['numbers_checked']} numbers checked)"]
    if record is not None:
        lines += ["", "## Technical appendix", "```json",
                  json.dumps({"to_c3": record["to_c3"], "to_c4": record["to_c4"]}, indent=2), "```"]
    return "\n".join(lines) + "\n"