# Round-12 docx sync (DD-054): fix collateral duplication from round-11 string
# replacements, effective-channel dimension form per DD-002 code convention,
# notation consistency (M=64 -> N_RIS = 64), acceptance-summary qualification,
# discussion quantifications, data-availability simplification.
import docx

SRC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
d = docx.Document(SRC)

def paros(fn):
    for i, p in enumerate(d.paragraphs):
        if fn(p.text):
            return i, p
    return None, None

# ---- P12 abstract: dedupe baseline -----------------------------------------
i, p = paros(lambda t: '(53.18 bps/Hz) (53.18 bps/Hz)' in t)
p.text = p.text.replace('(53.18 bps/Hz) (53.18 bps/Hz)', '(53.18 bps/Hz)')
print('P%d abstract deduped' % i)

# ---- P20 research-gap paragraph: full rewrite ------------------------------
i, p = paros(lambda t: 'solutionsnes in continuous spectrum' in t)
p.text = (
    'The RIS phase-control literature is still developing, and cross-paper comparison remains '
    'difficult: published methods are usually evaluated in their own simulation setups with distinct '
    'channel models, panel counts, reward definitions and training budgets, so reported improvements '
    'are rarely directly comparable. Two further gaps shape this work. First, widely used myopic '
    'heuristics search over fixed discrete phase grids rather than exploiting the full continuous '
    'phase space, which leaves headroom unquantified. Second, to our knowledge, no prior study has '
    'compared model-driven deep-unfolding approaches with model-free DRL policies under matched '
    'channel-state-error conditions and unseen user activity inside a single frozen test protocol, '
    'or weighed unfolding-controlled convergence against the flexibility of a model-free policy. '
    'This report addresses both gaps under one equal-parameter, shared-protocol evaluation.')
print('P%d research-gap paragraph rewritten' % i)

# ---- notation: M=64 -> N_RIS = 64 ------------------------------------------
i2, p2 = paros(lambda t: 'the M=64 dimensional phase vector' in t and 'MDP vocabulary' in t)
p2.text = p2.text.replace('the M=64 dimensional phase vector', 'the N_RIS = 64-dimensional phase vector')
i3, p3 = paros(lambda t: 'selects the M=64 dimensional phase vector' in t)
p3.text = p3.text.replace('selects the M=64 dimensional phase vector', 'selects the N_RIS = 64-dimensional phase vector')
i4, p4 = paros(lambda t: 'continuous M=64 phase space' in t)
p4.text = p4.text.replace('continuous M=64 phase space', 'continuous N_RIS = 64 phase space')
print('P%d/P%d/P%d bandit notation unified to N_RIS' % (i2, i3, i4))

# ---- P63 effective-channel dimension form (DD-002) -------------------------
i, p = paros(lambda t: 'effective-channel equation follows concatenation' in t)
k = p.text.find('Dimensionally,')
p.text = p.text[:k] + (
    'Dimensionally, h_d has shape K x N_BS, G is N_BS x N_RIS, h_r is K x N_RIS, and Theta is the '
    'N_RIS x N_RIS diagonal phase matrix. The effective channel is applied exactly as implemented '
    '(code convention DD-002): for each user k, h_eff_k = h_d_k + (h_r_k o e^{j theta}) G^H, i.e. the '
    'phases are applied elementwise to user k\'s RIS hop and then multiplied through the Hermitian '
    'transpose of G; with Theta diagonal the stacked form is H_eff = H_d + (h_r Theta) G^H, of shape '
    'K x N_BS with row h_eff_k in C^N_BS, matching h_d_k in C^N_BS. These shapes were verified '
    'line-by-line against src/precoder.py (zf_precoder, effective_channel). ' + p.text[p.text.find('.', k) + 2:])
print('P%d effective-channel dims rewritten per DD-002' % i)

# ---- P98 results: qualified null -------------------------------------------
i, p = paros(lambda t: 'No trained DRL architecture exceeds random' in t)
p.text = p.text.replace(
    'No trained DRL architecture exceeds random: the test-set deltas',
    'No trained DRL architecture showed a statistically significant positive gain over random: the '
    'test-set deltas')
print('P%d results null wording qualified' % i)

# ---- P118 discussion: dedupe repeated clause -------------------------------
i, p = paros(lambda t: 'a statistically significant positive mean gain:' in t)
p.text = p.text.replace(
    'showed a statistically significant positive mean gain: a statistically significant positive '
    'mean gain over random phase selection (53.18 bps/Hz)',
    'showed no statistically significant positive mean gain over random phase selection (53.18 bps/Hz)')
print('P%d discussion deduped' % i)

# ---- P122 discussion: quantified policy claim + latency grammar ------------
i, p = paros(lambda t: "Deep unfolding's decision latency stands at" in t)
p.text = p.text.replace(
    'every DRL policy evaluated here is below random in sum rate on the frozen test set while',
    'none of the evaluated DRL policies shows a statistically reliable positive gain over random in '
    'sum rate on the frozen test set (sample-level deltas remain within -0.10 to +0.04 bps/Hz) while')
p.text = p.text.replace(
    "Deep unfolding's decision latency stands at 8.26 ms (same-machine protocol, Section 5) is still "
    'only one component',
    'Deep unfolding\'s decision latency of 8.26 ms (same-machine protocol, Section 5) is only one '
    'component')
print('P%d network-design discussion fixed' % i)

# ---- P129 conclusion: acceptance qualification + dedupe + "does.." ---------
i, p = paros(lambda t: 'does.. No trained DRL policy' in t)
p.text = p.text.replace(
    'under one frozen 500-channel evaluation protocol whose acceptance criteria were frozen before '
    'test-set evaluation, with one documented post-hoc protocol revision: the AO-to-random ratio '
    'threshold was re-baselined from 1.30 to 1.15 after the conjugation fix invalidated the original '
    "calibration (disclosed in the report's protocol disclosure). The DRL-versus-random null and the "
    'unfolding-versus-AO rate ratio do not depend on that threshold; only the AO-gate pass/fail '
    'summary does.. No trained DRL policy showed a statistically significant positive mean gain over '
    'random phase selection (53.18 bps/Hz) (53.18 bps/Hz):',
    'under one frozen 500-channel evaluation protocol; its acceptance criteria were fixed before '
    'test-set evaluation except for one documented post-hoc revision, the AO-to-random ratio '
    'threshold re-baselined from 1.30 to 1.15 after the conjugation fix invalidated its original '
    'calibration. The DRL-versus-random null and the unfolding-versus-AO rate ratio do not depend '
    'on that threshold; only the AO-gate pass/fail summary does. No trained DRL policy showed a '
    'statistically significant positive mean gain over random phase selection (53.18 bps/Hz):')
print('P%d conclusion fixed' % i)

# ---- P135 data availability: simplify --------------------------------------
i, p = paros(lambda t: 'documents every protocol decision' in t)
p.text = p.text.replace(
    "the project decision log documents every protocol decision",
    "the full decision log is provided in docs/design_decisions.md")
print('P%d data availability simplified' % i)

d.save(SRC)

# ---- verification sweep ----------------------------------------------------
a = docx.Document(SRC)
jt = '\n'.join(q.text for q in a.paragraphs)
checks = {
    'dup (53.18) (53.18)': '(53.18 bps/Hz) (53.18 bps/Hz)' in jt,
    'dup clause': 'mean gain: a statistically significant' in jt,
    "does..": 'does..' in jt,
    'hereoices==mangled tail': 'solutionsnes' in jt,
    'M=64 residue': 'M=64' in jt,
    'exceeds random residue': 'exceeds random' in jt,
    'latency grammar residue': 'stands at 8.26 ms (same-machine protocol, Section 5) is' in jt,
    'decision-log redundancy': 'documents every protocol decision' in jt,
}
print('residues:', {k: v for k, v in checks.items() if v})
print('all clear:', not any(checks.values()))
