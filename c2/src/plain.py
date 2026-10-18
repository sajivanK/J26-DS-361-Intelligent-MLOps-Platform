"""
Plain-language decoding for the C2 page and business report.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

The model sees encoded data: continuous features are standardised (z-scores)
and categorical features are label-encoded integers. A business reader needs
"capital gain above about 5,150 and marital status is Married-civ-spouse",
not "capital_gain > 0.5518 and marital_status == 2".

The team preprocessor (shared/uci_adult/models/preprocessor.pkl, from Drive)
holds the fitted StandardScaler and LabelEncoders, so it decodes exactly.
If the file is missing, the decoder falls back to readable feature names and
the encoded values, and says so (decoded=False).

Usage:
    pl = PlainLanguage()
    pl.feature("capital_gain")                       # "capital gain"
    pl.rule("capital_gain > 0.5518 and marital_status == 2")
"""
import re
import sys

import joblib

from c2.src.config import SHARED_DIR

PREPROCESSOR_PKL = SHARED_DIR / "uci_adult" / "models" / "preprocessor.pkl"

NAMES = {
    "age": "age", "workclass": "type of employer", "education_num": "education level",
    "marital_status": "marital status", "occupation": "occupation",
    "relationship": "family role", "race": "race", "sex": "sex",
    "capital_gain": "capital gain", "capital_loss": "capital loss",
    "hours_per_week": "hours worked per week", "native_country": "country of origin",
}
UNITS = {"age": " years", "hours_per_week": " hours"}
CONTINUOUS = {"age", "education_num", "capital_gain", "capital_loss", "hours_per_week"}

_COND = re.compile(r"^\s*(\w+)\s*(>|<=|>=|<|==)\s*(-?[\d.]+|\w+)\s*$")
_OPS = {">": "above", ">=": "at least", "<": "below", "<=": "at most"}


def _load_preprocessor(path):
    """Load the team preprocessor; it may have been pickled from __main__."""
    from uci_adult.preprocessing import UCIAdultPreprocessor
    main = sys.modules["__main__"]
    had = hasattr(main, "UCIAdultPreprocessor")
    setattr(main, "UCIAdultPreprocessor", UCIAdultPreprocessor)
    try:
        return joblib.load(path)
    finally:
        if not had:
            delattr(main, "UCIAdultPreprocessor")


class PlainLanguage:

    def __init__(self, path=PREPROCESSOR_PKL):
        self.scale, self.classes, self.decoded = {}, {}, False
        try:
            pre = _load_preprocessor(path)
            cols = list(getattr(pre.scaler, "feature_names_in_", []))
            for i, c in enumerate(cols):
                self.scale[c] = (float(pre.scaler.mean_[i]), float(pre.scaler.scale_[i]))
            for c, le in pre.label_encoders.items():
                self.classes[c] = [str(x).strip() for x in le.classes_]
            self.decoded = True
        except Exception:  # noqa: BLE001  file missing or incompatible
            pass

    def feature(self, f):
        return NAMES.get(f, f.replace("_", " "))

    def value(self, f, v):
        """Encoded value -> original value as text."""
        v = float(v)
        if f in self.scale:
            mean, sd = self.scale[f]
            raw = mean + v * sd
            raw = round(raw / 10) * 10 if abs(raw) >= 1000 else round(raw)
            return f"{raw:,.0f}{UNITS.get(f, '')}"
        if f in self.classes and 0 <= int(v) < len(self.classes[f]):
            return self.classes[f][int(v)]
        if f in CONTINUOUS:                          # not decoded: say so
            return f"{v:g} (scaled value)"
        return f"code {int(v)}"

    def condition(self, cond):
        m = _COND.match(cond)
        if not m:
            return cond
        f, op, v = m.groups()
        if v == f:                                   # privacy fallback "f == f"
            return "every record"
        name = self.feature(f)
        if op == "==":
            return f"{name} is {self.value(f, v)}"
        return f"{name} {_OPS[op]} {'about ' if f in self.scale else ''}{self.value(f, v)}"

    def rule(self, rule):
        if not rule:
            return "no group"
        return " and ".join(self.condition(c) for c in rule.split(" and "))


if __name__ == "__main__":
    # Quick self-check: python3.11 -m c2.src.plain
    pl = PlainLanguage()
    print("decoded with preprocessor.pkl:", pl.decoded)
    for r in ["capital_gain > 0.5518 and marital_status == 2", "age <= -0.7763",
              "native_country == native_country"]:
        print(f"{r:<50} -> {pl.rule(r)}")