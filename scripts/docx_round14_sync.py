# Round-14 (DD-056): template-compliance + seed-selection reconciliation.
# Artifact-verified truth (results/baselines/zf_rescore/certified_table_det.json,
# every entry recorded with "seed": 0): the published DRL rows are the canonical
# SEED-0 checkpoints, deterministically re-scored under the corrected rate
# function; a validation-best-seed rule would have picked seeds 1/3, so both the
# docx 3.4 sentence and paper sec5c claims are corrected to the artifact truth.
import docx, re

DOC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
d = docx.Document(DOC)

def paros(fn):
    for i, p in enumerate(d.paragraphs):
        if fn(p.text):
            return i, p
    return None, None

# ---------------- 1) title: 13 words, docx + metadata -----------------------
TITLE = ("Model-Driven Optimization for Reconfigurable Intelligent Surfaces: "
         "Deep Unfolding versus Deep Reinforcement Learning")
i, p = paros(lambda t: t.startswith('Model-Driven Optimization for Reconfigurable'))
p.text = TITLE
d.core_properties.title = TITLE
d.core_properties.subject = 'RIS phase control comparison study'
d.core_properties.keywords = 'RIS; deep unfolding; deep reinforcement learning; null result; sum rate'
d.core_properties.comments = ('Companion repository: github.com/deathstarfall/ris-drl; '
                              'all numbers derive from committed artifacts.')
print('title set (%d words); metadata title updated' % len(TITLE.split()))

# ---------------- 2) abstract trim to ~215 ----------------------------------
i, p = paros(lambda t: t.startswith('This report evaluates direct deep-reinforcement-learning'))
ab = p.text
ab = ab.replace(
    ' All methods are trained and scored under one frozen protocol '
    '(1,500 training channels, 500 independently drawn frozen test channels), and every '
    'reported number derives from committed raw artifacts.',
    ' All methods share one frozen protocol (1,500 training channels, 500 frozen test '
    'channels), and every reported number derives from committed raw artifacts.')
ab = ab.replace(
    ' Rationale for every measurement choice and protocol decision is disclosed in the '
    'accompanying decision log; the study operates at 3.5 GHz with simplified i.i.d.-Gaussian '
    'fading and zero-forcing reception, so none of the conclusions extends outside this regime.',
    ' The study operates at 3.5 GHz with simplified i.i.d.-Gaussian fading and zero-forcing '
    'reception, with every protocol decision disclosed in the companion decision log.')
print('abstract now %d words' % len(ab.split()))
p.text = ab

# ---------------- 3) seed-selection reconciliation (P95) --------------------
i, p = paros(lambda t: 'the validation-best seed of each agent' in t)
p.text = p.text.replace(
    "Each DRL architecture was trained over five random seeds with an identical "
    "200,000-step budget, and the validation-best seed of each agent was evaluated on the test "
    "set (validation-best selection rule fixed before evaluation; no post-hoc selection, seed "
    "replacement, or metric change was permitted).",
    "Each DRL architecture was trained over five random seeds with an identical "
    "200,000-step budget. The published DRL rows are each architecture's canonical seed-0 "
    "checkpoint evaluated on the frozen test set, deterministically (fixed evaluation order, "
    "no exploration noise) under the corrected rate function; the per-channel rows, Wilcoxon "
    "p-values and 95% bootstrap intervals for these six checkpoints are stored in "
    "results/baselines/zf_rescore/certified_table_det.json (one entry per architecture, "
    "recording seed 0). Seed 0 was fixed as the certified checkpoint before the post-fix "
    "re-scoring; no post-hoc seed replacement or metric change was permitted. The remaining "
    "seed checkpoints (1-4) are retained as separate artifacts and are summarized, with their "
    "cross-seed spread, in the full-observation campaign of Table 12; the learning-rate "
    "selection for the unfolding network alone uses a validation split (below).")
print('P%d seed-selection reconciled to artifact truth' % i)

# ---------------- 4) Table 6 caption + pointing sentences -------------------
i, p = paros(lambda t: t.strip().startswith('Table 6:'))
p.text = ('Table 6: The three survey papers reviewed in this literature review ([17], [18], '
          '[19]), with their coverage, evaluation metrics and limitations.')
i46, p46 = paros(lambda t: 'Tables 1 to 5 span five research directions' in t)
if 'Table 6 summarizes the three surveys' not in p46.text:
    k = p46.text.find(" Table 6 summarizes")
    if k >= 0:
        j = p46.text.find('comparison', k)
        p46.text = (p46.text[:k] +
                    ' Table 6 tabulates the three surveys that motivated the gap analysis, '
                    'Table 7 compares representative prior studies with this report\'s '
                    'evaluation design, and the four research gaps are discussed above.')
print('Table 6 caption/pointers fixed')

# ---------------- 5) editorial: missing spaces, 'the paper', Table 8( -------
for p in d.paragraphs:
    p.text = p.text.replace('adopts.These', 'adopts. These')
    p.text = re.sub(r'([a-z])\.([A-Z][a-z])', lambda m: m.group(0) if m.group(2) == 'g' else
                    m.group(1) + '. ' + m.group(2), p.text)
    p.text = p.text.replace('matching Table 8(Training hyperparameters',
                            'matching Table 8 (Training hyperparameters')
    p.text = p.text.replace('as in the paper (a robustness sweep', 'as in the report (a '
                            'robustness sweep')
    p.text = p.text.replace("matching the paper's quantization table",
                            "matching the report's quantization table")
print('editorial spacing/paper-vs-report fixed')

d.save(DOC)
print('saved docx')

# ---------------- paper-side: same reconciliation ---------------------------
pa = r'C:\code\ris-drl\paper\sections\sec5c_certified_drl.tex'
c = open(pa, encoding='utf-8').read()
c = c.replace(
    'The validation-best seed of each agent was subsequently\nevaluated on the frozen '
    '$500$-channel test set through the shared\n\\texttt{evaluate.py} entry point '
    '(\\S\\ref{sec:results-ao}); all rates\nreport the same ZF sum-rate convention as every '
    'other row.',
    'The published DRL rows are each architecture\'s canonical\nseed-0 checkpoint, evaluated '
    'deterministically on the frozen $500$-channel test\nset through the shared '
    '\\texttt{evaluate.py} entry point (\\S\\ref{sec:results-ao});\nthe per-channel rows, '
    'Wilcoxon $p$-values and bootstrap intervals are\ncarried in the committed artifact '
    '\\texttt{certified\\_table\\_det.json} (one entry per\narchitecture, recording seed 0); '
    'all rates report the same ZF sum-rate convention as\nevery other row.')
c = c.replace('(validation-best seed per agent;', '(canonical seed-0 checkpoints per agent;')
open(pa, 'w', encoding='utf-8').write(c)
print('paper sec5c reconciled:', 'validation-best seed per agent' not in c)
