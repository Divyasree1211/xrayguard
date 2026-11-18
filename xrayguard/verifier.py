"""Clinical consistency verifier.

Parses a free-text report into finding mentions (with negation, hedging, side, zone and size words) and checks every
statement against the evidence layer. Also detects statements the evidence layer cannot support at all
(comparisons with non-existent priors, devices, measurements, out-of-vocabulary findings). A deterministic corrector
removes or rewrites failing sentences, so the final report is consistent with the evidence by construction.
"""
import re

from .config import LABELS
from .evidence import LATERAL, ZONED

FINDING_PATTERNS = {
    "Atelectasis": r"\batelecta\w*|\bvolume loss\b",
    "Cardiomegaly": (r"\bcardiomegaly\b|\bcardiac enlargement\b|\benlarge(?:d|ment of the) (?:cardiac silhouette|heart)\b"
                     r"|\b(?:cardiac silhouette|heart(?: size)?) (?:is |appears |may be |might be |is likely |is possibly |"
                     r"is probably )?(?:mildly |moderately |markedly |slightly )?enlarged\b"),
    "Effusion": r"\beffusions?\b|\bblunting of the (?:right |left |bilateral )?costophrenic\b|\bcostophrenic (?:angle )?blunting\b",
    "Infiltration": r"\binfiltrat\w*|\bairspace (?:opacit\w*|disease)\b",
    "Mass": r"\bmass(?:es)?\b",
    "Nodule": r"\bnodul\w*",
    "Pneumonia": r"\bpneumonia\b|\binfectious process\b",
    "Pneumothorax": r"\bpneumothora(?:x|ces)\b",
    "Consolidation": r"\bconsolidat\w*",
    "Edema": r"\bo?edema\b|\bvascular congestion\b",
    "Emphysema": r"\bemphysema\w*|\bhyperinflat\w*|\bhyperexpan\w*",
    "Fibrosis": r"\bfibros\w*|\bfibrotic\b|\bscarring\b",
    "Pleural_Thickening": r"\bpleural thickening\b|\bpleural plaques?\b|\bthickened pleura\b",
    "Hernia": r"\bhernia\w*",
}
FINDING_RE = {k: re.compile(v, re.I) for k, v in FINDING_PATTERNS.items()}
GENERIC_OPACITY = re.compile(r"\bopacit(?:y|ies)\b", re.I)
PARENCHYMAL = {"Atelectasis", "Mass", "Nodule", "Pneumonia", "Consolidation", "Edema", "Fibrosis", "Infiltration"}
CARDIO_NORMAL = re.compile(
    r"heart size is (?:within normal limits|normal)|normal (?:heart size|cardiac silhouette)|"
    r"(?:cardiac|cardiomediastinal) silhouette is (?:within normal limits|normal|not enlarged)|heart is not enlarged", re.I)
GLOBAL_NORMAL = re.compile(
    r"no acute (?:cardiopulmonary|intrathoracic|chest) (?:abnormalit\w*|process|disease|findings?)|"
    r"\blungs are clear\b|\bclear lungs\b|\bnormal chest\b|\bunremarkable (?:chest|study|radiograph)\b", re.I)

PRIOR_RE = re.compile(r"\b(?:compared (?:to|with)|comparison|prior|previous(?:ly)?|interval|unchanged|again (?:seen|noted|"
                      r"demonstrated)|stable|no change|worsen\w*|improv\w*|progression)\b", re.I)
DEVICE_RE = re.compile(r"\b(?:tubes?|catheters?|lines?|pacemakers?|pacer|wires?|sternotomy|surgical clips?|stents?|"
                       r"ports?|drains?|prosthe\w*|hardware|leads)\b", re.I)
MEASURE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:cm|mm|centimet\w*|millimet\w*)\b", re.I)
UNVERIFIABLE_RE = re.compile(r"\b(?:fractures?|scoliosis|kyphosis|granulomas?|calcifi\w*|lymphadenopathy|adenopathy|"
                             r"mediastinal widening|widened mediastinum|osteopenia|degenerative|free air|pneumoperitoneum|"
                             r"cavitat\w*|abscess\w*|tumou?rs?|cancer|malignan\w*|metasta\w*|tubercul\w*|covid\w*)\b", re.I)
SEVERITY_RE = re.compile(r"\b(?:small|mild|mildly|minimal|trace|tiny|moderate|moderately|large|severe|severely|massive|"
                         r"extensive|marked|markedly|slight|slightly|subtle)\b", re.I)

NEG_PRE = {"no", "not", "without", "negative", "free", "absence", "absent", "resolution", "resolved", "clear", "nor"}
NEG_POST = re.compile(r"^\W*(?:is|are|was|were)?\s*(?:not (?:seen|identified|present|evident|visualized|demonstrated)|"
                      r"absent|resolved|ruled out)\b", re.I)
HEDGE_PRE = {"possible", "possibly", "probable", "probably", "may", "might", "could", "likely", "suspicious", "suspected",
             "suggest", "suggests", "suggestive", "questionable", "equivocal", "concerning", "consider", "uncertain",
             "versus", "vs", "perhaps", "early", "borderline"}
HEDGE_POST = re.compile(r"^\W*(?:cannot be (?:excluded|ruled out)|can ?not be (?:excluded|ruled out)|not (?:be )?excluded|"
                        r"(?:is|are) (?:suspected|possible|likely|questioned|not excluded)|may be present|is not entirely excluded|\?)",
                        re.I)
TERMINATORS = re.compile(r"\b(?:but|however|although|though|whereas|except|which)\b|;", re.I)
TOKEN = re.compile(r"[A-Za-z]+|\?")
SIDE_WORDS = {"right": "right", "left": "left", "bilateral": "bilateral", "bilaterally": "bilateral", "both": "bilateral"}
ZONE_WORDS = {"upper": "upper", "apical": "upper", "apex": "upper", "apices": "upper", "mid": "mid", "middle": "mid",
              "lower": "lower", "base": "lower", "bases": "lower", "basal": "lower", "basilar": "lower"}

ERROR_TYPES = ["UNSUPPORTED_FINDING", "OVERCONFIDENT", "CONTRADICTION", "GLOBAL_CONTRADICTION", "LATERALITY",
               "UNSUPPORTED_SEVERITY", "PRIOR_REFERENCE", "UNSUPPORTED_DEVICE", "UNSUPPORTED_MEASUREMENT",
               "UNVERIFIABLE_FINDING", "UNSUPPORTED_NEGATIVE", "OMISSION"]
WARNING_TYPES = ["OMISSION_POSSIBLE", "ZONE"]
DROP_TYPES = {"UNSUPPORTED_FINDING", "OVERCONFIDENT", "CONTRADICTION", "GLOBAL_CONTRADICTION", "LATERALITY", "ZONE",
              "PRIOR_REFERENCE", "UNSUPPORTED_DEVICE", "UNSUPPORTED_MEASUREMENT", "UNVERIFIABLE_FINDING",
              "UNSUPPORTED_NEGATIVE"}


# ----------------------------------------------------------------------------- parsing
def split_report(text):
    """'FINDINGS: ... IMPRESSION: ...' -> (findings, impression). Tolerant to missing headers."""
    text = (text or "").strip()
    m = re.search(r"impression\s*:", text, re.I)
    if m:
        f, i = text[: m.start()], text[m.end():]
    else:
        f, i = text, ""
    f = re.sub(r"^\s*findings\s*:\s*", "", f.strip(), flags=re.I)
    return f.strip(), i.strip()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip(" .")]


def _clause_bounds(sent, start, end):
    lo, hi = 0, len(sent)
    for m in TERMINATORS.finditer(sent):
        if m.end() <= start:
            lo = max(lo, m.end())
        elif m.start() >= end:
            hi = min(hi, m.start())
            break
    return lo, hi


def _tokens(s):
    return [t.lower() for t in TOKEN.findall(s)]


def _mention_attrs(sent, start, end):
    lo, hi = _clause_bounds(sent, start, end)
    before = _tokens(sent[lo:start])
    after_txt = sent[end:hi]
    after = _tokens(after_txt)
    inside = _tokens(sent[start:end])
    pre7, pre5, post5 = before[-7:], before[-5:], after[:5]
    negated = any(t in NEG_PRE for t in pre7) or bool(NEG_POST.match(after_txt))
    if any(t in NEG_PRE for t in pre7) and "cannot" in pre7:
        negated = False
    hedged = any(t in HEDGE_PRE for t in pre7 + inside) or bool(HEDGE_POST.match(after_txt)) or "?" in after_txt[:20]
    side = None
    window = pre5 + post5
    if any(SIDE_WORDS.get(t) == "bilateral" for t in window) or re.search(r"\b(?:right and left|left and right)\b",
                                                                          " ".join(pre5)):
        side = "bilateral"
    else:
        for t in reversed(pre5):
            if t in ("right", "left"):
                side = t
                break
        if side is None:
            for t in post5:
                if t in ("right", "left"):
                    side = t
                    break
    zone = next((ZONE_WORDS[t] for t in reversed(pre5) if t in ZONE_WORDS), None) or \
        next((ZONE_WORDS[t] for t in post5 if t in ZONE_WORDS), None)
    sev_txt = " ".join(before[-4:] + inside + after[:3])
    sev = SEVERITY_RE.findall(sev_txt)
    return dict(negated=negated, hedged=hedged, side=side, zone=zone, severity=[s.lower() for s in sev])


def parse_sentence(sent):
    """All finding mentions in one sentence."""
    out = []
    for lab, rx in FINDING_RE.items():
        for m in rx.finditer(sent):
            a = _mention_attrs(sent, m.start(), m.end())
            out.append(dict(label=lab, text=m.group(0), span=(m.start(), m.end()), **a))
    if not any(o["label"] in PARENCHYMAL for o in out):
        for m in GENERIC_OPACITY.finditer(sent):
            a = _mention_attrs(sent, m.start(), m.end())
            out.append(dict(label="Infiltration", text=m.group(0), span=(m.start(), m.end()), **a))
    for m in CARDIO_NORMAL.finditer(sent):
        out.append(dict(label="Cardiomegaly", text=m.group(0), span=(m.start(), m.end()), negated=True, hedged=False,
                        side=None, zone=None, severity=[]))
    # one mention per label per sentence: prefer a positive one (e.g. "No effusion; right effusion" is odd anyway)
    best = {}
    for o in out:
        k = o["label"]
        if k not in best or (best[k]["negated"] and not o["negated"]):
            best[k] = o
    return list(best.values())


def _scope_hits(rx, sent):
    hits = []
    for m in rx.finditer(sent):
        a = _mention_attrs(sent, m.start(), m.end())
        if not a["negated"]:
            hits.append(m.group(0))
    return hits


def parse_report(findings, impression):
    """-> list of sentence records: {section, idx, text, mentions, extras}"""
    recs = []
    for sec, txt in (("findings", findings), ("impression", impression)):
        for i, s in enumerate(split_sentences(txt)):
            recs.append(dict(section=sec, idx=i, text=s, mentions=parse_sentence(s),
                             prior=PRIOR_RE.findall(s), device=_scope_hits(DEVICE_RE, s),
                             measure=MEASURE_RE.findall(s), unverifiable=_scope_hits(UNVERIFIABLE_RE, s),
                             global_normal=bool(GLOBAL_NORMAL.search(s))))
    return recs


def text_to_labels(text):
    """Report text -> {label: 'definite' | 'hedged' | 'negative' | None} (rule-based labeller used for evaluation)."""
    f, i = split_report(text)
    res = {n: None for n in LABELS}
    for r in parse_report(f, i):
        for m in r["mentions"]:
            cur = res[m["label"]]
            if m["negated"]:
                if cur is None:
                    res[m["label"]] = "negative"
            else:
                val = "hedged" if m["hedged"] else "definite"
                if cur in (None, "negative") or (cur == "hedged" and val == "definite"):
                    res[m["label"]] = val
    return res


# ----------------------------------------------------------------------------- verification
def verify_report(findings, impression, evidence, cfg=None):
    rcfg = (cfg or {}).get("report", {})
    allow_extent = rcfg.get("allow_extent", False)
    check_zone = rcfg.get("check_zone", True)
    issues, mentioned = [], set()
    recs = parse_report(findings, impression)
    any_present = [n for n, d in evidence.items() if d["tier"] == "present"]
    any_possible = [n for n, d in evidence.items() if d["tier"] == "possible"]

    def add(t, lab, detail, r):
        issues.append(dict(type=t, label=lab, detail=detail, section=r["section"] if r else None,
                           idx=r["idx"] if r else None, sentence=r["text"] if r else None))

    for r in recs:
        for m in r["mentions"]:
            lab, ev = m["label"], evidence[m["label"]]
            tier = ev["tier"]
            if m["negated"]:
                if tier == "present":
                    add("CONTRADICTION", lab, f"report denies {lab} but evidence p={ev['prob']:.2f} (present)", r)
                elif tier == "possible" and not m["hedged"]:
                    add("OVERCONFIDENT", lab, f"report rules out {lab} but evidence is uncertain (p={ev['prob']:.2f})", r)
                elif tier == "absent":
                    add("UNSUPPORTED_NEGATIVE", lab, f"a below-threshold score does not establish absence of {lab}", r)
                continue
            mentioned.add(lab)
            if tier == "absent":
                add("UNSUPPORTED_FINDING", lab, f"{lab} not supported by evidence (p={ev['prob']:.2f})", r)
                continue
            if tier == "possible" and not m["hedged"]:
                add("OVERCONFIDENT", lab, f"{lab} stated as definite but evidence is only 'possible' (p={ev['prob']:.2f})", r)
            if lab in LATERAL and m["side"] and ev.get("side") and m["side"] != ev["side"]:
                add("LATERALITY", lab, f"report says {m['side']}, evidence says {ev['side']}", r)
            if check_zone and lab in ZONED and m["zone"] and ev.get("zone") and m["zone"] != ev["zone"]:
                add("ZONE", lab, f"report says {m['zone']} zone, evidence says {ev['zone']}", r)
            if m["severity"] and not allow_extent:
                add("UNSUPPORTED_SEVERITY", lab, f"size/severity '{' '.join(m['severity'])}' is not measured", r)
        if r["global_normal"]:
            if any_present:
                add("GLOBAL_CONTRADICTION", None, f"'normal' statement but present: {', '.join(any_present)}", r)
            elif any_possible:
                add("OVERCONFIDENT", None, f"'normal' statement but possible: {', '.join(any_possible)}", r)
            else:
                add("UNSUPPORTED_NEGATIVE", None, "below-threshold scores do not establish a normal examination", r)
        if r["prior"]:
            add("PRIOR_REFERENCE", None, f"refers to a prior study ({', '.join(r['prior'])}); none was provided", r)
        if r["device"]:
            add("UNSUPPORTED_DEVICE", None, f"device not assessed by the model ({', '.join(r['device'])})", r)
        if r["measure"]:
            add("UNSUPPORTED_MEASUREMENT", None, f"measurement not available ({', '.join(r['measure'])})", r)
        if r["unverifiable"]:
            add("UNVERIFIABLE_FINDING", None, f"finding outside the evidence vocabulary ({', '.join(r['unverifiable'])})", r)

    for n, d in evidence.items():
        if n not in mentioned:
            if d["tier"] == "present":
                add("OMISSION", n, f"{n} present (p={d['prob']:.2f}) but not reported", None)
            elif d["tier"] == "possible":
                add("OMISSION_POSSIBLE", n, f"{n} possible (p={d['prob']:.2f}) but not reported", None)
    return issues


def errors_only(issues):
    return [i for i in issues if i["type"] in ERROR_TYPES]


def feedback_text(issues):
    return "\n".join(f"- {i['type']}: {i['detail']}" for i in errors_only(issues))


# ----------------------------------------------------------------------------- correction
def correct_report(findings, impression, evidence, cfg=None):
    """Remove/rewrite failing sentences, strip unmeasured size words, add omitted findings from the template."""
    from .report import finding_sentence, impression_phrase, template_report
    issues = verify_report(findings, impression, evidence, cfg)
    log = []
    bad = {}
    for i in issues:
        if i["section"] is not None:
            bad.setdefault((i["section"], i["idx"]), set()).add(i["type"])
    sections = {}
    for sec, txt in (("findings", findings), ("impression", impression)):
        kept = []
        for k, s in enumerate(split_sentences(txt)):
            types = bad.get((sec, k), set())
            if types & DROP_TYPES:
                log.append(f"removed ({'/'.join(sorted(types))}): \"{s}\"")
                continue
            if "UNSUPPORTED_SEVERITY" in types:
                s2 = re.sub(r"\s{2,}", " ", SEVERITY_RE.sub("", s)).replace(" ,", ",").strip()
                s2 = re.sub(r"\b(a|an)\s+(?=[aeiou])", "an ", s2)
                s2 = re.sub(r"\ban\s+(?=[^aeiou\W])", "a ", s2)
                s2 = s2[:1].upper() + s2[1:]
                log.append(f"removed size/severity words: \"{s}\" -> \"{s2}\"")
                s = s2
            kept.append(s if s.endswith((".", "?", "!")) else s + ".")
        sections[sec] = kept

    def mentioned_ok(sents):
        ok = set()
        for s in sents:
            for m in parse_sentence(s):
                if not m["negated"]:
                    ok.add(m["label"])
        return ok

    need = [n for n, d in evidence.items() if d["tier"] in ("present", "possible")]
    have_f = mentioned_ok(sections["findings"])
    for n in need:
        if n not in have_f:
            sections["findings"].append(finding_sentence(n, evidence[n]))
            log.append(f"added {evidence[n]['tier']} finding to FINDINGS: {n}")
    if not sections["findings"]:
        sections["findings"] = split_sentences(template_report(evidence)[0])
        log.append("FINDINGS empty after correction -> template findings")
    have_i = mentioned_ok(sections["impression"])
    missing_i = [n for n in need if n not in have_i]
    if not sections["impression"] or missing_i:
        if not need:
            if not sections["impression"]:
                sections["impression"] = [template_report(evidence)[1]]
                log.append("IMPRESSION empty -> template impression")
        else:
            order = sorted(missing_i, key=lambda n: (evidence[n]["tier"] != "present", -evidence[n]["prob"]))
            sections["impression"] += [impression_phrase(n, evidence[n]) for n in order]
            log.append(f"added to IMPRESSION: {', '.join(order)}")
    f2, i2 = " ".join(sections["findings"]), " ".join(sections["impression"])
    remaining = errors_only(verify_report(f2, i2, evidence, cfg))
    return dict(findings=f2, impression=i2, log=log, issues_before=issues, remaining=remaining)
