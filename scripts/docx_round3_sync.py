# round-3 docx sync (DD-045). Anchored to live file content verified 2026-10-27.
import docx, shutil, os

SRC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
BAK = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report_pre_round3_sync.docx'
shutil.copyfile(SRC, BAK)
d = docx.Document(SRC)

def sub_in_para(p, old, new):
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new); return True
    if old in p.text:
        p.text = p.text.replace(old, new); return True
    return False

def find(sub, start=0):
    for i, p in enumerate(d.paragraphs[start:], start):
        if sub in p.text: return i, p
    return None, None

# 1) affiliation typo + corresponding author with name ------------------------
i, p = find('School of Electronic Engineering')
if p: sub_in_para(p, 'School of Electronic Engineering', 'School of Electronics Engineering')
i, p = find('Corresponding author: 24BDS0166')
if p:
    p.text = ('Corresponding author: Ayansh Agarwal (Team 6, project lead). '
              'Contact: github.com/deathstarfall/ris-drl (issues); '
              'institutional e-mail finalized at journal submission.')

# 2) abstract: de-6G sentence --------------------------------------------------
i, p = find('one of the main technical components of the future 6G')
if p:
    p.text = ('Reconfigurable Intelligent Surface (RIS) is an array of passive reflecting devices '
              'that modifies the propagation environment in order to improve network performance '
              'in terms of signal throughput, coverage and energy efficiency. ' +
              p.text[p.text.find('.')+1:].lstrip())

# 3) intro: neutral 5G/6G data-rate paragraph ----------------------------------
i, p = find('5G has very fast data')
if p:
    set_text = p
    p.text = ('After the worldwide rollout of 5G, attention has shifted toward the next]'
              .replace(']', '') or '')
# careful: do it as a proper rewrite
i, p = find('5G has very fast data')
if p:
    p.text = ('Following the 5G rollout, industry groups are defining a next-generation (broadly '
              '"6G") agenda. This report deliberately does not rest any claim on future-generation '
              'performance targets: it evaluates RIS phase control inside one fixed mid-band '
              'simulation regime and with one frozen evaluation protocol, and every performance '
              'statement is tied to committed artifacts rather than to generation narratives.')

# 4) leftover MDP paragraph (methods voice) -------------------------------------
i, p = find('posed as a Markov decision process')
if p:
    p.text = ('Each decision episode is one step: the policy observes the current channel '
              'observation, selects the M=64 dimensional phase vector (continuous or a 2, 3 or '
              '4-bit codebook), receives the zero-forcing sum rate as its reward, and the episode '
              'terminates immediately. Because feedback is one-shot, there is no discounting and '
              'no credit assignment; the exploration noise and the replay buffer exist only to '
              'stabilize function approximation, not to propagate value across decisions.')

# 5) merge the two 3.5 GHz disclaimers (keep the second, drop the first) ---------
i1, p1 = find('framed for a mid-band 3.5 GHz')
i2, p2 = find('motivated by 6G-class RIS deployments')
if i1 is not None and i2 is not None and i1 < i2:
    p2.text = p2.text  # keep
    p1._element.getparent().remove(p1._element)

# 6) internal jargon in prose ----------------------------------------------------
for pat, new in (('certified null result', 'measured null result'),
                 ('certified-null', 'verified null'),
                 ('certified null', 'verified null'),
                 ('pre-registered', 'fixed in advance'),
                 ('DD-001 through DD-044', 'the project decision log')):
    i, p = find(pat)
    while p is not None:
        sub_in_para(p, pat, new)
        i, p2 = find(pat, (i or 0) + 1)
        p = p2

# 7) G2 relaxation disclosure + modest margin statement ---------------------------
i, p = find('64.193 +/- 0.582')
if p and '1.30' not in p.text:
    p.add_run(' Acceptance-criteria disclosure (auditor round 3): the margin criterion was set at '
        'ratio >= 1.30 before testing; the conjugation fix invalidated the calibration scale '
        '(it was computed under the erroneous rate), and the criterion was re-baselined to '
        'ratio >= 1.15 with 100% per-channel dominance before reading the corrected result - '
        'a disclosed post-hoc change, logged as DD-043. Under either criterion AO beats random '
        'on 100% of the frozen channels, but the margin itself is modest (1.21x), which is '
        'why the headline of this study is the unfolding-vs-random gain, not the AO margin.')

# 8) latency-protocol statement ---------------------------------------------------
i, p = find('1044x')
if p and 'serialized' not in p.text:
    p.add_run(' Latency protocol note: every per-channel latency figure in this report is a '
        'serialized single-solve wall clock measured the same way on one desktop CPU; the AO '
        'run\'s "total wall" line (96.7 s) is a parallel-elapsed figure across 4 worker '
        'processes and is not a per-channel number. The retired ~9.6 ms unfolding figure '
        'which appears in early drafts was a float32 development-stage benchmark on different '
        'hardware and is disclosed in the decision log (DD-037) as retired, never published.')

# 9) drop "Crossref-verified" from table captions ----------------------------------
lead = d.paragraphs
for p in d.paragraphs:
    if 'Crossref-verified' in p.text and p.text.strip().lower().startswith('table'):
        sub_in_para(p, 'Crossref-verified ', '')
        sub_in_para(p, 'Crossref-verified', '')
for t in d.tables:
    for r in t.rows:
        for c in r.cells:
            for cp in c.paragraphs:
                if 'Crossref-verified' in cp.text:
                    sub_in_para(cp, 'Crossref-verified', '')

d.save(SRC)
print('saved')
chk = d.Document if False else None
import docx as dx2
again = dx2.Document(SRC)
txt = '\n'.join(q.text for q in again.paragraphs)
tbl = '\n'.join(' | '.join(c.text for c in r.cells) for t in again.tables for r in t.rows)
allt = txt + tbl
for k, v in {
    'Markov gone': 'Markov decision process' not in txt,
    '6G-components gone': 'technical components of the future 6G' not in txt,
    '5G-fast gone': '5G has very fast data' not in txt,
    'one 3.5GHz para': txt.count('3.5 GHz') == 2,
    'certified-null gone': 'certified null' not in allt,
    'pre-registered gone': 'pre-registered' not in allt,
    'DD-00 gone': 'DD-00' not in allt,
    'Crossref-verified gone': 'Crossref-verified' not in tbl,
    'Electronics fixed': 'School of Electronics Engineering' in txt,
    'G2 disclosure in': '1.30' in txt,
    'latency note in': 'serialized' in txt,
}.items():
    print(k, 'OK' if v else 'FAIL')
