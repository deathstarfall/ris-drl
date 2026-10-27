# Round-11 docx sync (DD-053): "no positive mean gain" -> "no statistically significant
# positive gain" everywhere; dimension symbols N_BS/N_RIS; accurate propagation-model
# description (DD-042 single cascade attenuation, metres); acceptance-protocol wording;
# Table 10 moved before the conclusion; Table 11 statistical definitions; "herethere"
# paragraph rewrite; data-availability redundancy; "Deep unfolding's an 8.26 ms" grammar.
import docx
import numpy as np

SRC = r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx'
d = docx.Document(SRC)
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

def find(sub, start=0):
    for i, p in enumerate(d.paragraphs[start:], start):
        if sub in p.text:
            return i, p
    return None, None

NULL_OLD = 'achieved a positive mean gain over random phase selection'
NULL_NEW = ('showed a statistically significant positive mean gain over random phase selection '
            '(53.18 bps/Hz)')
NULL_OLD2 = 'achieved a positive mean gain over random'
# 1) null wording (abstract P12, P116, P127)
for sub in ('achieved a positive mean gain', 'No trained DRL policy achieved a positive mean gain',
            'no DRL architecture (six architectures, five seeds each, 200,000-step budgets) achieved a positive'):
    p = find(sub)[1]
    if p is None:
        continue
    p.text = p.text.replace(
        'achieved a positive mean gain over random phase selection',
        'showed a statistically significant positive mean gain over random phase selection (53.18 bps/Hz)')
    p.text = p.text.replace(
        'achieved a positive mean gain over random',
        'showed a statistically significant positive mean gain over random')
    print('null wording fixed:', sub[:50])
# P116 discussion: also make the sample-means point explicit
i116, p116 = find('The measured outcome is asymmetric')
if p116 is not None:
    p116.text = p116.text.replace(
        'Under the frozen protocol no DRL architecture (six architectures, five seeds each, 200,000-step budgets) showed',
        'Under the frozen protocol no DRL architecture (six architectures, five seeds each, 200,000-step budgets) showed a statistically significant positive mean gain:')
    if 'small positive and negative sample means' not in p116.text:
        p116.text += (' Wording note: small positive and negative sample means do occur (SAC +0.036, '
                      'DQN-4-bit +0.026 in the full-observation stress test); none of them is statistically '
                      'distinguishable from the random baseline under the reported bootstrap and paired '
                      'Wilcoxon tests, which is the statement the null makes - equivalence cannot be proven, '
                      'only that no reliable positive effect was detected.')
        print('P116 wording note appended')

# 2) dimension symbols + 3) accurate propagation description (P76)
i76, p76 = find('The experiments use synthetic channels')
if p76 is not None:
    p76.text = (
        'The experiments use synthetic channels for a downlink scenario with N_BS = 8 base-station antennas, '
        'K = 4 single-antenna users and an RIS with N_RIS = 64 passive elements, with a blocked direct link '
        '(implementation note: the released code stores the same dimensions as M = 8 for the BS antennas and '
        'N = 64 for the RIS elements; every matrix below is dimensioned accordingly, [h_d] in C^(K x N_BS), '
        '[G] in C^(N_BS x N_RIS), [h_r] in C^(K x N_RIS)). All channel entries are circularly symmetric complex '
        'Gaussian (Rayleigh fading), drawn i.i.d. and scaled by 3GPP urban-micro path-loss models, described '
        'exactly as simulated: the direct link uses the NLoS UMi formula 128.1 + 37.6 log10(d_km) dB with the '
        'transmit-power link budget P = 30 dBm; the cascade link uses a SINGLE joint two-hop attenuation in '
        'the Wu-Zhang form, 61.4 + 35.8 log10(max(d_BS-RIS, d_RIS-user)) dB with distances in METRES, i.e. the '
        'attenuation covers the worse of the two hops; this shared factor is profiled onto the RIS-to-user hop '
        'h_r while the BS-to-RIS matrix G carries unit magnitude, so that the shared loss is not double-counted. '
        'This is a deliberate simplification disclosed in the project decision log; it implies the generator '
        'does not resolve per-hop propagation, and frequency-dependent mmWave or sub-THz effects are outside '
        'its scope. The noise power follows the full link budget from -174 dBm/Hz with B = 1 MHz and NF = 7 dB '
        '(sigma^2 = -107 dBm). The evaluation uses a frozen set of 500 independently drawn test channels; the '
        'DRL policies and the unfolding network are trained on the frozen training split (1,500 training channels).'
        ' Each policy observes a four-part stack flattened to 1,056 real values per decision (2 x 256 from the '
        'magnitude and angle of h_r, 512 magnitudes from G, and 32 magnitudes from h_d); crucially this stack '
        'contains the magnitudes and the angle of h_r but NOT the angles of G or h_d, on which the optimal phases '
        'depend (quantified in Section 5). The only optimization variable is the vector of 64 RIS reflection '
        'phases; the zero-forcing sum rate is the reward.')
    print('P76 system model rewritten (dims + propagation)')
# P59 constants consistency with Eq (1)/(2)
i59, p59 = find('The base station has N antennas')
if p59 is not None:
    p59.text = ('The base station has N_BS = 8 antennas while K = 4 single-antenna users receive their signal '
                'through an RIS with N_RIS = 64 passive components. The phase-shift matrix of the surface is '
                'written in the code convention as Theta = diag(e^{j theta_1}, ..., e^{j theta_m}, ...), where '
                'the implementation indexes the same 64 elements as N; the equations below use the element '
                'index m = 1, ..., N_RIS throughout.')
    print('P59 dims fixed')
# Eq (1)/(2) subscripts -> N_RIS
ipe, pe = find('θ_M}),  θ_m ∈ [0, 2π)')
if pe is None:
    ipe, pe = find('e^{jθ_M}')
if pe is not None:
    pe.text = pe.text.replace('e^{jθ_M}),  θ_m ∈ [0, 2π),  m = 1, …, M', 'e^{jθ_{N_RIS}}),  θ_m ∈ [0, 2π),  m = 1, …, N_RIS')
    print('Eq (1) symbols fixed')
i62, p62 = find('max over Θ')
if p62 is None:
    i62, p62 = find('SINR_k(Θ)')
if p62 is not None:
    p62.text = ('max over Θ:   R(Θ) = Σ_{k=1..K} log₂(1 + SINR_k(Θ)),   subject to |[Θ]_{m,m}| = 1 for all '
                'm = 1, …, N_RIS     --(2)')
    print('Eq (2) symbols fixed')
i61, p61 = find('Θ is the diagonal matrix of RIS phase shifts')
if p61 is not None:
    p61.text = p61.text.replace('θ_m is the phase of the m-th element and M is the number of surface components',
                                'θ_m is the phase of the m-th element, m = 1, ..., N_RIS')
    print('Eq(1) definition aligned')
# Eq definition sentence with dims
i63, p63 = None, None
for i, p in enumerate(d.paragraphs):
    if p.text.startswith('In Eq. (2) the sum runs over the K users'):
        i63, p63 = i, p
if p63 is not None:
    p63.text = ('In Eq. (2) the sum runs over the K = 4 users and SINR_k(Θ) is user k\'s achievable SINR under '
                'the zero-forcing precoder at the base station with total transmit power P; no per-user '
                'power-splitting heuristic is used. Dimensionally, Θ is N_RIS x N_RIS, the effective '
                'BS-user channels form a K x N_BS matrix, and the effective-channel equation follows '
                'concatenation h_eff = h_d + G Θ h_r with the sizes above. The control problem is a one-step '
                'contextual bandit (T = 1): each episode picks one channel, one phase vector is chosen, the sum '
                'rate is observed, and the episode terminates. Because every episode ends after a single '
                'decision, there is no bootstrap term and no discounting.')
    print('Eq(2) definition + dims')

# 4) acceptance-protocol wording (conclusion)
i127, p127 = find('fixed in advance acceptance rules')
if p127 is None:
    i127, p127 = find('fixed-in-advance acceptance rules')
if p127 is not None:
    p127.text = p127.text.replace(
        'under one frozen 500-channel evaluation protocol with fixed in advance acceptance rules',
        'under one frozen 500-channel evaluation protocol whose acceptance criteria were frozen before test-set '
        'evaluation, with one documented post-hoc protocol revision: the AO-to-random ratio threshold was '
        're-baselined from 1.30 to 1.15 after the conjugation fix invalidated the original calibration (disclosed '
        'in the report\'s protocol disclosure). The DRL-versus-random null and the unfolding-versus-AO rate ratio '
        'do not depend on that threshold; only the AO-gate pass/fail summary does.')
    print('acceptance wording fixed')
else:
    print('MISS acceptance phrase')

# 6) move Table 10 (quantization) + its caption + interpretation before the conclusion
cap10 = find('Table 10  Post-hoc B-bit phase quantization')[1]
p131 = find('The quantization losses are 2.936')[1]
tbl10el = None
t9el = None
for t in d.tables:
    hdr = ' '.join(c.text for c in t.rows[0].cells)
    if 'Method' in hdr and '2-bit (4 levels)' in hdr:
        tbl10el = t._tbl
    if 'channel-state error' in hdr.lower() or 'eps' in hdr.lower() and 'AO' in hdr:
        t9el = t._tbl
# fall back: use the CSI discussion paragraph (P104) as anchor
anchor = find('Channel-state error was injected as h_hat')[1]
if tbl10el is not None and cap10 is not None and p131 is not None:
    for el in (cap10._p, tbl10el, p131._p):
        el.getparent().remove(el)
    e = anchor._p if t9el is None else t9el
    for el in (cap10._p, tbl10el, p131._p):
        e.addnext(el)
        e = el
    print('Table 10 block moved into Results (after CSI block)')
else:
    print('MISS move (cap10/tbl10/p131)', cap10 is not None, tbl10el is not None, p131 is not None)

# 7) Table 11 caption statistical definitions
for p in d.paragraphs:
    if p.text.startswith('Table 11'):
        k = p.text.find('random floor 53.178.')
        p.text = (p.text[:k] +
                  'random floor 53.178. Each +/- value is the sample SD (n-1) across the FIVE training seeds of '
                  'the per-seed mean test rate (averaged over the 500 paired channels inside each seed). The '
                  'Wilcoxon tests are paired on the seed-averaged per-channel rates, i.e. 500 paired '
                  'observations, one pair per frozen channel; the five-seed DRL campaign equally applies: '
                  'family-level statements rest on five training seeds per architecture and cannot rule out '
                  'rare-seed effects. ' + p.text[k + len('random floor 53.178.'):])
        print('Table 11 stats clarified')
        break

# 8) malformed 'hereoices' paragraph rewrite + verification wording accuracy
ip20, p20 = find('no one tested new user activity')
if p20 is None:
    i20, p20 = find('hereoices of fixed phases')
ip20b, p20b = find('hereoices of fixed phases')
if p20b is None:
    p20b = find('hereoices')[1]
if p20 is not None:
    p20.text = p20.text.replace(
        'RIS still is a topic which is being actively explored. Most papers compare their new approach only to one another like DDPG. Also each of them used unique simulation setup, channel model, number of panels, reward function and training time. So any reported improvements for convolution based DDPG can not be considered comparable. There are many unresolved questions in this field as well. To our knowledge, these families have not been compared in one equal-parameter, shared-protocol evaluation covering the practical regimes considered here',
        'The RIS phase-control literature is still developing, and cross-paper comparison remains difficult: '
        'published methods are usually evaluated in their own simulation setups with distinct channel models, '
        'panel counts, reward definitions and training budgets, so reported improvements are rarely directly '
        'comparable. Several questions therefore remain open in this field. To our knowledge, these method '
        'families have not been compared under one equal-parameter, shared-protocol evaluation of the kind used here')
    print('P20 comparison paragraph rewritten')
if p20b is not None:
    p20b.text = p20b.text.replace('hereoices of fixed phases rather than theoretically optimal o',
                                  'over fixed discrete phase grids rather than theoretically optimal continuous solutions')
    if 'hereoices' in p20b.text:
        k = p20b.text.find('hereoices')
        ctx = p20b.text[max(0, k - 80):k + 90]
        p20b.text = p20b.text.replace('hereoices', 'phase-selection heuristics')
        print('hereoices context-part fixed:', ctx[:80])
# verification wording: DOI vs arXiv
for p in d.paragraphs:
    if 'verified against Crossref publisher metadata - DOI resolution plus title match' in p.text:
        p.text = p.text.replace(
            'All 19 bibliographic entries have been verified against Crossref publisher metadata - DOI resolution plus title match - for the sixteen corpus studies and the three surveys alike; entries that could not be verified were dropped entirely rather than annotated.',
            'All 19 bibliographic entries are verified by the repository checker: DOI resolution plus title match for the sixteen entries that carry a DOI, and metadata verification through the arXiv/PMLR identifiers for the three DOI-less entries (the SAC, DDPG and Mnih conference records). Entries that could not be verified were dropped entirely rather than annotated.')
        print('verification wording made exact')

# 9) grammar + data-availability redundancy
for p in d.paragraphs:
    if "Deep unfolding's an 8.26 ms decision latency" in p.text:
        p.text = p.text.replace("Deep unfolding's an 8.26 ms decision latency",
                                "Deep unfolding's decision latency stands at 8.26 ms")
        print('grammar fixed')
    if 'the decision log documents the project decision log' in p.text:
        p.text = p.text.replace('the decision log documents the project decision log',
                                'the project decision log documents every protocol decision')
        print('data-availability deduped')

d.save(SRC)
a = docx.Document(SRC)
allp = '\n'.join(q.text for q in a.paragraphs)
print('verify: hereoices gone:', 'hereoices' not in allp,
      '| stat-sig wording:', allp.count('statistically significant positive'),
      '| residual bare PMG:', allp.count('positive mean gain over random phase selection.)') ,
      '| unconditional gone:', 'unconditional' not in allp)
