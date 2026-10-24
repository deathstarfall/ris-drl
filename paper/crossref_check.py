"""Reference verification v3: DOI-first, PMLR-aware, arXiv-aware.
  doi present            -> resolve exact DOI on Crossref, compare titles
  eprint + arXiv prefix  -> check arXiv API title for that id
  PMLR series + url      -> fetch url, require the title on the page
  none of the above      -> Crossref title search, must score >= 0.90 and not be a repository record
  PLACEHOLDER in title   -> hard FAIL
"""
import re, sys, html, difflib, requests

BIBFILE = "paper/refs.bib"
REPO_HOSTS = ("techrxiv", "arxiv", "f1000", "10356", "10.3204", "osf")
THRESH = 0.90

def parse_bib(text):
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        kind, key = m.group(1).lower(), m.group(2)
        if kind in ("comment", "string", "preamble"):
            continue
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body, fields, pos = text[m.end():i - 1], {}, 0
        while True:
            fm = re.compile(r"(\w+)\s*=\s*").search(body, pos)
            if not fm:
                break
            j = fm.end()
            if body[j] == "{":
                d, k = 1, j + 1
                while k < len(body) and d:
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                val = body[j + 1:k - 1]
            elif body[j] == '"':
                k = body.index('"', j + 1) + 1
                val = body[j + 1:k - 1]
            else:
                k = body.find(",", j)
                k = len(body) if k < 0 else k
                val = body[j:k].strip()
            fields[fm.group(1).lower()] = val
            pos = k
        yield key, kind, fields

def clean(t): return re.sub(r"\s+", " ", re.sub(r"[{}\\]", "", t)).strip()
def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", clean(t).lower())).strip()
def ratio(a, b): return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()

ok, bad, warn = [], [], []
text = open(BIBFILE, encoding="utf-8").read()
for key, kind, f in parse_bib(text):
    title = clean(f.get("title", ""))
    if not title:
        bad.append((key, "no title field")); continue
    if "PLACEHOLDER" in title.upper():
        bad.append((key, "placeholder entry: fill or delete")); continue
    doi = clean(f.get("doi", "")).replace("https://doi.org/", "", 1)
    eprint = clean(f.get("eprint", ""))
    series = f.get("series", "")
    url = clean(f.get("url", ""))
    try:
        if doi:
            r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=20)
            if r.status_code != 200:
                bad.append((key, f"DOI {doi} not found on Crossref (HTTP {r.status_code})")); continue
            item = r.json()["message"]
            s = ratio(title, (item.get("title") or [""])[0])
            if s < 0.80: bad.append((key, f"DOI {doi} resolves to a DIFFERENT paper (score {s:.2f})"))
            else: ok.append((key, f"DOI verified: {doi} (score {s:.2f})"))
        elif eprint and "arxiv" in f.get("archiveprefix", "").lower():
            x = requests.get("https://export.arxiv.org/api/query", params={"id_list": eprint}, timeout=20).text
            m = re.search(r"<entry>.*?<title>(.*?)</title>", x, re.S)
            s = ratio(title, m.group(1)) if m else 0
            if s < 0.80: bad.append((key, f"arXiv:{eprint} title mismatch (score {s:.2f})"))
            else: ok.append((key, f"arXiv:{eprint} verified (score {s:.2f})"))
        elif "machine learning research" in series.lower():
            if not url:
                bad.append((key, "PMLR entry needs url = {https://proceedings.mlr.press/...}")); continue
            page = html.unescape(requests.get(url, timeout=20).text)
            if norm(title) in norm(page): ok.append((key, f"PMLR page title verified: {url}"))
            else: bad.append((key, f"title not found on {url}"))
        else:
            hit = requests.get("https://api.crossref.org/works",
                               params={"query.bibliographic": title, "rows": 1},
                               timeout=20).json()["message"]["items"][0]
            s = ratio(title, (hit.get("title") or [""])[0])
            d = clean(hit.get("DOI", "?"))
            if s < THRESH: bad.append((key, f"weak match {s:.2f} -> {d}: add doi/eprint/url"))
            elif any(h in (d + str(hit.get("URL", ""))).lower() for h in REPO_HOSTS):
                bad.append((key, f"matches repository record {d}: add doi or eprint"))
            else: ok.append((key, f"title search: {d} (score {s:.2f}); consider adding doi"))
    except Exception as e:
        bad.append((key, f"network/parse error: {e}"))

for k, m in ok: print(f" OK  {k:26s} {m}")
if bad:
    print("\nFAILURES:")
    for k, m in bad: print(f"  - {k}: {m}")
print("\nRESULT:", "ALL ENTRIES VERIFIED ✓" if not bad else f"{len(bad)} entries blocked ✗")
sys.exit(1 if bad else 0)