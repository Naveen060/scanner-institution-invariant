"""Prose checks on the compiled manuscript (not part of the scientific pipeline).

Usage: python scripts/text_checks.py paper/ieee/main.pdf [--out runs/text_checks.txt]
Extracts the running text (abstract .. conclusion), then reports
  - LanguageTool (offline, en-US) grammar/style matches, filtered for maths noise
  - readability (Flesch reading ease, FK grade) and sentence-length spread
  - hedge-word and number density
  - the longest repeated word 4-grams (a cheap proxy for self-plagiarism/boilerplate)
"""
import re, subprocess, sys, collections, statistics

pdf = sys.argv[1]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
raw = subprocess.run(["pdftotext", "-layout", "-nopgbrk", pdf, "-"], capture_output=True, text=True).stdout
# two-column layout: use raw reading order instead
raw = subprocess.run(["pdftotext", "-nopgbrk", pdf, "-"], capture_output=True, text=True).stdout
i = raw.find("Index Terms")
j = raw.find("ACKNOWLEDGMENT")
body = raw[i:j if j > 0 else len(raw)]
body = re.sub(r"-\n(?=[a-z])", "", body)
paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", body)]
# drop table/figure debris: paragraphs that are short or number-dominated
keep = []
for p in paras:
    if len(p.split()) < 25: continue
    digits = sum(c.isdigit() for c in p) / max(1, len(p))
    if digits > 0.12: continue
    if p.startswith(("TABLE", "Fig.")): continue
    keep.append(p)
text = "\n".join(keep)
lines = []
def say(s=""):
    print(s); lines.append(s)

words = re.findall(r"[A-Za-z']+", text)
say(f"running text: {len(words)} words, {len(keep)} paragraphs")

def syl(w):  # vowel-group heuristic (textstat's dictionary needs an NLTK download)
    w = w.lower(); n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1: n -= 1
    return max(1, n)
nsent = max(1, len(re.findall(r"[.!?](?:\s|$)", text)))
sy = sum(syl(w) for w in words)
fre = 206.835 - 1.015 * len(words) / nsent - 84.6 * sy / len(words)
fk = 0.39 * len(words) / nsent + 11.8 * sy / len(words) - 15.59
say(f"Flesch reading ease {fre:.1f}; Flesch-Kincaid grade {fk:.1f} (vowel-group syllable heuristic)")
sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z$\\])", text) if len(s.split()) > 2]
L = [len(s.split()) for s in sents]
say(f"sentences {len(L)}: mean {statistics.mean(L):.1f} words, sd {statistics.pstdev(L):.1f}, >35 words: {sum(l > 35 for l in L)}, <8 words: {sum(l < 8 for l in L)}")
hedges = ["may", "might", "could", "suggest", "suggests", "suggestive", "seem", "seems", "appear", "appears", "probably", "likely", "unlikely", "possibly", "somewhat", "roughly", "about"]
c = collections.Counter(w.lower() for w in words)
say("hedge words per 1000: %.1f  (%s)" % (1000 * sum(c[h] for h in hedges) / len(words), ", ".join(f"{h}:{c[h]}" for h in hedges if c[h])))
starts = collections.Counter(s.split()[0] for s in sents)
say("most common sentence openers: " + ", ".join(f"{w}:{n}" for w, n in starts.most_common(6)))
tr = collections.Counter(w.lower() for w in re.findall(r"\b(?:however|therefore|moreover|furthermore|additionally|thus|hence|notably|importantly)\b", text, flags=re.I))
say("connector words: " + ", ".join(f"{w}:{n}" for w, n in tr.most_common()))

# repeated 4-grams
toks = [w.lower() for w in words]
ng = collections.Counter(tuple(toks[k:k + 5]) for k in range(len(toks) - 4))
rep = [(g, n) for g, n in ng.most_common(40) if n >= 3]
say("5-grams repeated >=3 times: " + "; ".join(" ".join(g) + f" x{n}" for g, n in rep[:12]))

try:
    import language_tool_python
    tool = language_tool_python.LanguageTool("en-US")
    ms = tool.check(text)
    skip = {"MORFOLOGIK_RULE_EN_US", "WHITESPACE_RULE", "COMMA_PARENTHESIS_WHITESPACE", "UPPERCASE_SENTENCE_START", "EN_QUOTES", "DASH_RULE"}
    kept = [m for m in ms if m.rule_id not in skip]
    say(f"LanguageTool: {len(ms)} raw matches, {len(kept)} after dropping spelling/whitespace noise")
    for m in kept[:80]:
        ctx = text[max(0, m.offset - 40): m.offset + m.error_length + 40].replace("\n", " ")
        say(f"  [{m.rule_id}] {m.message[:90]} | ...{ctx}...")
    sp = [m for m in ms if m.rule_id == "MORFOLOGIK_RULE_EN_US"]
    bad = collections.Counter(text[m.offset:m.offset + m.error_length] for m in sp)
    say("spelling flags (mostly technical terms): " + ", ".join(f"{w}:{n}" for w, n in bad.most_common(25)))
    tool.close()
except Exception as e:
    say(f"LanguageTool unavailable: {e}")
try:
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(text); tmp = fh.name
    r = subprocess.run(["proselint", "check", tmp], capture_output=True, text=True)
    os.unlink(tmp)
    flags = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
    say(f"proselint: {len(flags)} output lines")
    for l in flags[:40]:
        say("  " + l[:200])
except Exception as e:
    say(f"proselint unavailable: {e}")
if out:
    open(out, "w").write("\n".join(lines))
