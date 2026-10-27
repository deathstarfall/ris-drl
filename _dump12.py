import docx
d = docx.Document(r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx')
targets = [
    (12, 'abstract'),
]
for i, p in enumerate(d.paragraphs):
    t = p.text
    hits = []
    if '(53.18 bps/Hz) (53.18 bps/Hz)' in t or '53.18 bps/Hz) (53.18' in t: hits.append('DUP53')
    if 'a statistically significant positive mean gain:' in t: hits.append('DUPCLAUSE')
    if 'does..' in t or 'does. No trained DRL' in t: hits.append('DOESDOT')
    if 'single fair comparison' in t: hits.append('GAPPARA')
    if 'phase-vector' in t or 'M = 64' in t or 'M=64' in t: hits.append('NOTATION')
    if "Deep unfolding's decision latency stands" in t: hits.append('LATENCYP')
    if 'documents every protocol decision' in t: hits.append('DATAAV')
    if 'frozen before test-set evaluation' in t: hits.append('FROZEN')
    if 'below random in sum rate' in t or 'below the random baseline in sum rate' in t: hits.append('BELOWRAND')
    if 'exceeds random' in t or 'exceed random' in t: hits.append('EXCEEDS')
    if 'G Theta h_r' in t or 'G Θ h_r' in t or 'G Θ h' in t: hits.append('HEFFEQ')
    if 'h_eff' in t or 'effective BS-user' in t: hits.append('HEFFDEF')
    if hits:
        print('P%d %s | %r' % (i, ','.join(hits), t[:400]))
        print()
