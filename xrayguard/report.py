"""Report generation from the evidence layer.

* TemplateGenerator - deterministic, evidence-constrained structured report (always available, no GPU needed).
* LLMGenerator      - an instruction-tuned language model writes the report from the evidence table; the verifier then
                      checks it, asks for one regeneration with feedback, and finally corrects it deterministically.
"""
import re

from .evidence import LATERAL, ZONED, evidence_table

NAME = {
    "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Effusion": "pleural effusion",
    "Infiltration": "pulmonary infiltrate", "Mass": "mass", "Nodule": "nodule", "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax", "Consolidation": "consolidation", "Edema": "pulmonary edema",
    "Emphysema": "emphysema", "Fibrosis": "fibrosis", "Pleural_Thickening": "pleural thickening",
    "Hernia": "hiatal hernia",
}
COUNTABLE = {"Effusion", "Infiltration", "Mass", "Nodule", "Pneumothorax", "Hernia"}
PLURAL = {"Effusion": "pleural effusions", "Infiltration": "pulmonary infiltrates", "Mass": "masses",
          "Nodule": "nodules", "Pneumothorax": "pneumothoraces"}
ORDER = ["Consolidation", "Pneumonia", "Infiltration", "Atelectasis", "Mass", "Nodule", "Edema", "Fibrosis",
         "Emphysema", "Effusion", "Pneumothorax", "Pleural_Thickening", "Cardiomegaly", "Hernia"]


def _loc(n, d):
    if n not in LATERAL or not d.get("side"):
        return ""
    if d["side"] == "bilateral":
        return "bilateral "
    if n in ZONED and d.get("zone"):
        return f"{d['side']} {d['zone']} zone "
    return f"{d['side']} "


def _np(n, d):
    """noun phrase with location, plural for bilateral countable findings."""
    loc = _loc(n, d)
    if d.get("side") == "bilateral" and n in PLURAL:
        return f"{loc}{PLURAL[n]}", True
    return f"{loc}{NAME[n]}", False


def finding_sentence(n, d):
    phrase, plural = _np(n, d)
    hedge = d["tier"] == "possible"
    if n == "Cardiomegaly":
        return "There is possible cardiomegaly." if hedge else "The cardiac silhouette is enlarged, consistent with cardiomegaly."
    if n == "Emphysema":
        return "There are possible changes of emphysema." if hedge else "There are changes of emphysema."
    if plural:
        return f"There are possible {phrase}." if hedge else f"There are {phrase}."
    art = "a " if n in COUNTABLE else ""
    return f"There is {art}possible {phrase}." if hedge else f"There is {art}{phrase}."


def impression_phrase(n, d):
    phrase, _ = _np(n, d)
    p = f"Possible {phrase}." if d["tier"] == "possible" else f"{phrase[0].upper()}{phrase[1:]}."
    return p


def template_report(evidence):
    pos = [n for n in ORDER if evidence[n]["tier"] in ("present", "possible")]
    if not pos:
        return ("All 14 finding scores are below their reporting thresholds.",
                "Model assessment is inconclusive.")
    findings = [finding_sentence(n, evidence[n]) for n in pos]
    if any(d["tier"] == "absent" for d in evidence.values()):
        findings.append("Other finding scores are below their reporting thresholds. "
                        "A below-threshold score does not rule out disease.")
    pres = [n for n in pos if evidence[n]["tier"] == "present"]
    poss = [n for n in pos if evidence[n]["tier"] == "possible"]
    impression = " ".join(impression_phrase(n, evidence[n]) for n in pres + poss)
    return " ".join(findings), impression


# ----------------------------------------------------------------------------- LLM generator
SYSTEM_PROMPT = (
    "You are a radiology reporting assistant. You write chest X-ray reports strictly from a structured evidence "
    "table produced by an image model. You never add information that is not in the table.")

USER_TEMPLATE = """Evidence table (probabilities are calibrated; STATUS present = confident, possible = uncertain):
{table}

Write the report following these rules:
1. Report every finding with STATUS present as a definite finding, with its location if one is given.
2. Report every finding with STATUS possible using hedging words such as "possible" or "cannot be excluded".
3. Do not mention any finding that is not in the table. Do not mention devices, measurements, sizes or severity,
   and do not compare with prior studies.
4. Findings omitted from the table are below threshold. Do not state they are absent, rule out disease,
   or describe the examination or heart size as normal. A below-threshold score does not rule out disease.
5. Use exactly this format:
FINDINGS: <2-5 sentences>
IMPRESSION: <1-3 short sentences>"""

FEEDBACK_TEMPLATE = """Your previous report was:
{draft}

An automatic checker found these problems:
{feedback}

Rewrite the report so that it fixes every problem and still follows all the rules. Use the same format."""


class LLMGenerator:
    def __init__(self, model_name, device="cuda", max_new_tokens=220):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.device, self.max_new = device, max_new_tokens

    def _chat(self, user_msgs):
        return [self.tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                              {"role": "user", "content": u}],
                                             tokenize=False, add_generation_prompt=True) for u in user_msgs]

    def generate(self, user_msgs):
        import torch
        prompts = self._chat(user_msgs)
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id, repetition_penalty=1.05)
        return [self.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]


def llm_user_message(evidence):
    return USER_TEMPLATE.format(table=evidence_table(evidence))


def clean_llm_text(text):
    text = re.sub(r"[*#`]+", "", text or "")
    return re.sub(r"[ \t]+", " ", text).strip()


def generate_verified(evidences, llm, cfg, verify_fn, correct_fn, split_fn):
    """Generate -> verify -> (regenerate with feedback) -> deterministic correction, batched."""
    from .verifier import errors_only, feedback_text
    drafts = [clean_llm_text(t) for t in llm.generate([llm_user_message(e) for e in evidences])]
    current = list(drafts)
    regen_count = [0] * len(evidences)
    for _ in range(cfg["report"].get("max_regen", 1)):
        todo = []
        for k, (txt, ev) in enumerate(zip(current, evidences)):
            f, i = split_fn(txt)
            iss = errors_only(verify_fn(f, i, ev, cfg))
            if iss:
                todo.append((k, FEEDBACK_TEMPLATE.format(draft=txt, feedback=feedback_text(iss))))
        if not todo:
            break
        outs = llm.generate([llm_user_message(evidences[k]) + "\n\n" + fb for k, fb in todo])
        for (k, _), o in zip(todo, outs):
            current[k] = clean_llm_text(o)
            regen_count[k] += 1
    results = []
    for k, (txt, ev) in enumerate(zip(current, evidences)):
        f, i = split_fn(txt)
        fixed = correct_fn(f, i, ev, cfg)
        results.append(dict(draft=drafts[k], regenerated=current[k], regen=regen_count[k], **fixed))
    return results
