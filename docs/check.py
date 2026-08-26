"""Consistency checks over the generated documents.

Three failure modes this catches, all of which have actually happened here:

  1. a number typed into the text that duplicates a generated macro, which then
     silently stops matching when the experiment is re-run;
  2. a macro used in a document but never defined, which LaTeX renders as an
     undefined control sequence and a reader sees as a gap;
  3. a figure referenced by a document that does not exist on disk.

Exit code is non-zero if anything in category 2 or 3 is found. Category 1 is
reported as a warning, because a few literals are legitimately constants.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.expanduser("~/dev/isonavi")
DOCS = os.path.join(ROOT, "docs")
METRICS = os.path.join(DOCS, "metrics.tex")

# Numbers that are genuinely fixed, not results: physical constants, standard
# definitions, counts that define the design rather than measure it.
ALLOWED = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "16", "20",
    "24", "45", "50", "100", "150", "180", "200", "250", "1000", "2000",
    "1957", "2016", "2021", "2026", "35", "66", "0.5", "1.5", "2.4",
}


def load_macros():
    macros = {}
    for line in open(METRICS, encoding="utf-8"):
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line.strip())
        if m:
            macros[m.group(1)] = m.group(2)
    return macros


def body_of(path):
    """Document text with the preamble and comments removed."""
    s = open(path, encoding="utf-8").read()
    i = s.find(r"\begin{document}")
    if i > 0:
        s = s[i:]
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.splitlines())



# Headline numbers quoted in the readme, pinned to the macro that generates
# them. The readme is plain text, so nothing else stops it going stale while
# the experiments move underneath it.
README_PINS = [
    ("hydroEnvCad", "2.69"),
    ("siteCurrent", "2.4"),
    ("misPath", "573"),
    ("misDuration", "893"),
    ("repCoverMean", "95.3"),
    ("repNavMeanMean", "0.322"),
    ("repNavMeanStd", "0.161"),
    ("repDvlMean", "96.0"),
    ("repMapRmseMean", "0.285"),
    ("repDetErrMean", "1.80"),
    ("repDetErrStd", "0.40"),
    ("yoloTestMapFifty", "99.0"),
    ("hilNavMean", "0.107"),
    ("hilTicks", "17,884"),
    ("boardComputeMean", "34.6"),
    ("cadMass", "28.0"),
    ("cadNetBuoy", "1.96"),
    ("cadBG", "25.6"),
    ("structDepth", "50"),
    ("bomTotal", "38,27,800"),
    ("bomLakh", "38.3"),
    ("budgetProgramme", "86,73,412"),
    ("budgetProgrammeLakh", "86.7"),
    ("budgetCapital", "11,40,890"),
    ("budgetMaterials", "2,87,016"),
    ("budgetValidation", "14,91,984"),
    ("budgetManpower", "11,16,000"),
    ("budgetContingency", "8,09,723"),
    ("bomRate", "95.64"),
]


def check_readme(macros):
    """Every pinned readme figure must equal the macro that produces it."""
    path = os.path.join(ROOT, "README.md")
    if not os.path.exists(path):
        return []
    body = open(path, encoding="utf-8").read()
    bad = []
    for name, quoted in README_PINS:
        generated = macros.get(name)
        if generated is None:
            bad.append(f"README pin \\{name} has no generated macro")
        elif generated != quoted:
            bad.append(f"README pin \\{name}: readme expects {quoted}, "
                       f"metrics.tex now says {generated}")
        elif quoted not in body:
            bad.append(f"README no longer quotes {quoted} for \\{name}")
    return bad


def main():
    macros = load_macros()
    print(f"{len(macros)} generated macros")

    docs = [p for p in ("isonavi_report.tex", "isonavi_brief.tex")
            if os.path.exists(os.path.join(DOCS, p))]
    problems, warnings = [], []
    used = set()

    for name in docs:
        path = os.path.join(DOCS, name)
        body = body_of(path)

        for mac in re.findall(r"\\(\w+)", body):
            if mac in macros:
                used.add(mac)

        # 2. Macros that look generated but are not defined.
        for mac in set(re.findall(r"\\(\w+)", body)):
            if mac in macros:
                continue
            if re.match(r"^(mis|rep|det|cad|struct|hydro|bom|en|red|esp|"
                        r"board|hil|yolo|synth|xd|de|scour|map|site|sim)"
                        r"[A-Z]\w*$", mac):
                problems.append(f"{name}: \\{mac} looks generated but is "
                                f"not defined in metrics.tex")

        # 3. Figures that do not exist.
        for fig in re.findall(r"\{(\.\./[\w/\.-]+\.png)\}", body):
            if not os.path.exists(os.path.join(DOCS, fig)):
                problems.append(f"{name}: missing figure {fig}")

        # 1. Literals that duplicate a macro value.
        for lit in set(re.findall(r"(?<![\w.\\])(\d+\.\d+|\d+)(?![\w.])",
                                  body)):
            if lit in ALLOWED:
                continue
            hits = sorted(k for k, v in macros.items() if v == lit)
            if hits:
                warnings.append(f"{name}: literal {lit} also available as "
                                f"\\{hits[0]}")

    problems.extend(check_readme(macros))

    unused = sorted(set(macros) - used)

    if warnings:
        print(f"\n{len(warnings)} literal(s) duplicating a generated value:")
        for w in sorted(warnings)[:25]:
            print("  " + w)

    if unused:
        print(f"\n{len(unused)} macro(s) generated but unused "
              f"(harmless, listed for pruning):")
        print("  " + ", ".join(unused[:20])
              + (" ..." if len(unused) > 20 else ""))

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  " + p)
        return 1

    print("\nno undefined macros, no missing figures, readme in step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
