
import argparse
import pandas as pd
import re
from collections import Counter
from pathlib import Path

# lowercase keys map to canonical
CANONICAL_MAP = {
    # common palladium catalysts
    "pd(pph3)4": "Pd(PPh3)4",
    "pd(pph3)2cl2": "Pd(PPh3)2Cl2",
    "pd2(dba)3": "Pd2(dba)3",
    "pd2(dba)3·ch": "Pd2(dba)3",
    "pd": "Pd",
    # platinum
    "[pt]": "Pt",
    "pt": "Pt",
    # copper
    "cui": "CuI",
    "cubr": "CuBr",
    "cucl2": "CuCl2",
    # nickel
    "nicl2": "NiCl2",
    "ni": "Ni",
    # simple inorganic bases
    "[k+]": "K+",
    "k+": "K+",
    "[na+]": "Na+",
    "na+": "Na+",
    "cl": "Cl",
    "cl-": "Cl",
    # hydride
    "[h-]": "H-",
    "h-": "H-",
    # generic ligands (examples)
    "p(tbu)3": "P(tBu)3",
    "p(p h)3": "PPh3",
    # common textual forms
    "triphenylphosphine": "PPh3",
    "triethylamine": "Et3N",
}

# regexes for cleaning
RE_CHARGE = re.compile(r'[\[\]]')  # remove [] brackets
RE_NON_ALPHANUM = re.compile(r'[^0-9a-z\+\-\(\)\.]')  # keep some punctuation

def canonicalize_agent(raw: str) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if s == "" or s == "\\N":
        return ""
    s_low = s.lower()
    # remove surrounding whitespace and brackets
    s_low = RE_CHARGE.sub('', s_low)
    # collapse whitespace
    s_low = re.sub(r'\s+', ' ', s_low)
    # try direct map
    k = s_low.strip()
    if k in CANONICAL_MAP:
        return CANONICAL_MAP[k]
    # strip lots of non-alphanum (but keep + - () .)
    k2 = RE_NON_ALPHANUM.sub('', k)
    if k2 in CANONICAL_MAP:
        return CANONICAL_MAP[k2]
    # heuristics: if contains 'pd' and 'pph3'
    if 'pd' in k and 'pph3' in k:
        return "Pd(PPh3)4"
    if 'pd' in k:
        return "Pd"
    if 'pt' in k:
        return "Pt"
    if 'cu' in k:
        # CuI or Cu depending
        if 'i' in k:
            return "CuI"
        if 'br' in k:
            return "CuBr"
        return "Cu"
    # simple ionic forms like K+, Na+
    if re.search(r'\bk\b', k):
        return "K+"
    if re.search(r'\bna\b', k):
        return "Na+"
    # fallback return original string trimmed (but cleaned of trailing punctuation)
    return s.strip()

def normalize_file(input_csv: str, output_csv: str, agent_col: str = "agent_000"):
    df = pd.read_csv(input_csv, dtype=str, low_memory=False)
    # ensure column exists
    if agent_col not in df.columns:
        print(f"Warning: {agent_col} not in CSV columns. Creating empty column.")
        df[agent_col] = None
    # apply canonicalization
    df['agent_norm'] = df[agent_col].apply(canonicalize_agent)

    # produce a small report
    cnt_raw = Counter([str(x).strip() for x in df[agent_col].fillna('') if str(x).strip() != '' and str(x) != '\\N'])
    cnt_norm = Counter([x for x in df['agent_norm'].fillna('') if x != ''])
    report_lines = []
    report_lines.append(f"Input file: {input_csv}")
    report_lines.append(f"Rows: {len(df)}")
    report_lines.append(f"Unique raw agents (non-empty): {len(cnt_raw)}")
    report_lines.append(f"Unique normalized agents (non-empty): {len(cnt_norm)}")
    report_lines.append("\nTop 20 raw agents (count):")
    for a, c in cnt_raw.most_common(20):
        report_lines.append(f"  {a} : {c}")
    report_lines.append("\nTop 20 normalized agents (count):")
    for a, c in cnt_norm.most_common(20):
        report_lines.append(f"  {a} : {c}")

    # write outputs
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    report_path = str(output_csv) + ".report.txt"
    Path(report_path).write_text("\n".join(report_lines), encoding='utf-8')
    print(f"Wrote normalized CSV: {output_csv}")
    print(f"Wrote report: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path (normalized)")
    parser.add_argument("--agent-col", default="agent_000", help="Column name in CSV containing agent")
    args = parser.parse_args()
    normalize_file(args.input, args.output, agent_col=args.agent_col)
