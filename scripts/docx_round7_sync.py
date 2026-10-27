# Round-7 docx sync (DD-049): de-duplicate title, stray ").;", strip DD codes from body,
# SAC lr prose fix, Table 7 caption + Table 6 author-name sync, unified verification wording,
# testbed citation narrowed, batch-1 autograd latency note.
import docx

SRC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
d = docx.Document(SRC)

def find(sub, start=0):
    for i, p in enumerate(d.paragraphs[start:], start):
        if sub in p.text:
            return i, p
    return None, None

TITLE = ('Model-Driven Optimization for Reconfigurable Intelligent Surfaces: Deep Unfolding versus a '
         'Null Result for Direct Reinforcement Learning')
p0 = d.paragraphs[0]
if p0.text != TITLE:
    # re-write the whole paragraph onto its FIRST run's formatting
    if p0.runs:
        p0.runs[0].text = TITLE
        for r in p0.runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p0.text = TITLE
    print('title deduplicated ->', p0.text[:70])

# ---- abstract stray ").;" -----------------------------------------------------
i12, pa = find('an untrained forward pass')
if pa is not None and ').; an untrained' in pa.text:
    pa.text = pa.text.replace('disclosed).; an untrained', 'disclosed). An untrained')
    print('abstract ").;" fixed')

# ---- strip DD codes from body prose (keep only the reproducibility subsection) --
# locate the reproducibility subsection start so its DD mentions stay
rep_idx = None
for i, p in enumerate(d.paragraphs):
    if 'reproducib' in p.text.lower() and 'Decision log' in p.text:
        rep_idx = i
for i, p in enumerate(d.paragraphs):
    if rep_idx is not None and i >= rep_idx:
        break  # keep everything from the reproducibility section on
    if 'DD-04' in p.text:
        p.text = p.text.replace('DD-048 same-machine idle protocol', 'same-machine idle protocol')
        p.text = p.text.replace('DD-048 same-machine protocol', 'same-machine protocol')
        p.text = p.text.replace(' (DD-048)', '')
        p.text = p.text.replace(' (DD-048).', '.')
    if 'DD-0' in p.text:
        print('still has DD code: P%d' % i, p.text[:80])
for t in d.tables:
    if rep_idx is None or True:  # tables have no safe subsection; the only codes are body-adjacent
        for r in t.rows:
            for c in r.cells:
                if 'DD-048 same-machine idle protocol' in c.text:
                    c.text = c.text.replace('DD-048 same-machine idle protocol', 'same-machine idle protocol')
                    print('cell DD stripped')
                elif 'DD-0' in c.text:
                    print('cell still has DD:', c.text[:60])

# ---- SAC lr prose (Section 3.3) ------------------------------------------------
i82, p82 = find('lr 3e-4 for actor, critic and temperature')
if p82 is None:
    i82, p82 = find('3e-4 for actor, critic')
if p82 is not None:
    p82.text = p82.text.replace('lr 3e-4 for actor, critic and temperature',
                                'actor and temperature lr 3e-4, critic lr 1e-3 (as in Table 7 and the released configs)').replace(
                                'initial temperature 0.2; actor and temperature lr 3e-4, critic lr 1e-3 (as in Table 7 and the released configs)',
                                'initial temperature 0.2; actor and temperature lr 3e-4, critic lr 1e-3, matching Table 7 and the released configs')
    print('SAC prose lr fixed:', 'critic lr 1e-3' in p82.text)

# ---- Table 7 caption -----------------------------------------------------------
i90, p90 = find('Table 7 Training hyperparameters')
if p90 is not None:
    p90.text = ('Table 7 Training hyperparameters and configuration for the five evaluated agent '
                'families plus deep unfolding (columns: DDPG, TD3, SAC, codebook DQN at 2/3/4-bit, '
                'and unfolding; budgets are environment steps; discounting is inert because every '
                'episode is a single decision, so no TD bootstrap exists).')
    print('Table 7 caption updated')

# ---- Table 6 author-cell sync --------------------------------------------------
for t in d.tables:
    for r in t.rows:
        for c in r.cells:
            if c.text.startswith('F. Puspitasari, I. Lee'):
                c.text = c.text.replace('F. Puspitasari, I. Lee', 'A. A. Puspitasari, B. M. Lee')
                print('Puspitasari cell fixed')
            if c.text.startswith('A. Faisal, W. Choi'):
                c.text = c.text.replace('A. Faisal, W. Choi', 'K. M. Faisal, W. Choi')
                print('Faisal cell fixed')
            if c.text.startswith('M. Tariq, S. Ahmad, M. Ahmad Jan, et al.'):
                c.text = c.text.replace('M. Tariq, S. Ahmad, M. Ahmad Jan, et al.',
                                        'M. Tariq, S. Ahmad, M. A. Jan, H. Song')
                print('Tariq cell fixed')

# ---- unified verification wording ----------------------------------------------
ipar = None
for i, p in enumerate(d.paragraphs):
    if 'verified against Crossref (a DOI- or publisher-level match)' in p.text:
        ipar = i
for t in d.tables:
    for r in t.rows:
        for c in r.cells:
            if 'all entries checked against the original papers' in c.text:
                c.text = c.text.replace('all entries checked against the original papers',
                                        'all entries verified against Crossref publisher metadata')
                print('Table 1 caption unified')
for p in d.paragraphs:
    if p.text.strip().startswith('Table ') and 'all entries checked against the original papers' in p.text:
        p.text = p.text.replace('all entries checked against the original papers',
                                'all entries verified against Crossref publisher metadata')
        print('caption paragraph unified')

# ---- testbed citation narrows to the indoor-testbed paper ----------------------
i17, p17 = find('blocking-rich indoor links [8], [17]')
if p17 is not None:
    p17.text = p17.text.replace('demonstrated rate gains in blocking-rich indoor links [8], [17], but',
                                'demonstrated rate gains in blocking-rich indoor links [8], but')
    print('testbed citation narrowed to [8]')

# ---- batch-1 autograd overhead note --------------------------------------------
i118, p118 = find('A protocol control passes')
if p118 is not None:
    p118.text += (' One protocol note: the 8.26 ms decision cost is dominated by batch-1 autograd '
                  'overhead over six lightweight matrix steps on a small network; batched evaluation '
                  'amortizes that overhead, and the serialized per-channel figures are the honest '
                  'single-solve protocol measured identically for every method.')
    print('autograd note added')

d.save(SRC)
# verify
a = docx.Document(SRC)
allp = '\n'.join(q.text for q in a.paragraphs)
print('DONE. title dup gone:', allp.count('LearningModel-Driven'), '| ").;":', allp.count(').;'))
