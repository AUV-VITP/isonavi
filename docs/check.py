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

ROOT = os.path.expanduser("~/dev/rakshatech")
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


def main():
    macros = load_macros()
    print(f"{len(macros)} generated macros")

    docs = [p for p in ("varuna_report.tex", "varuna_brief.tex")
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

    print("\nno undefined macros, no missing figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
