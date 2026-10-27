import docx, re
d = docx.Document(r'C:\Users\user\OneDrive\Desktop\All Assignments\CN project\TeamNo_6_24BDS0166_RIS_6G_Report.docx')
allp = [p.text for p in d.paragraphs]
print('--- captions ---')
for i, t in enumerate(allp):
    if re.match(r'^(Table|Figure) \d+', t.strip()):
        print('P%d |%s|' % (i, t[:130]))
print('--- in-text Table/Figure mentions (non-caption) ---')
for i, t in enumerate(allp):
    if not re.match(r'^(Table|Figure) \d', t.strip()):
        for m in re.finditer(r'(Table|Figure) (\d+)[\s\n]', t):
            print('P%d cites %s %s' % (i, m.group(1), m.group(2)))
print('--- headings ---')
for i, t in enumerate(allp):
    s = t.strip()
    if re.match(r'^\d+\.\s+[A-Z]', s) or s.lower().startswith(('declarations', 'references', 'keywords')):
        print('P%d |%s|' % (i, s[:60]))
print('--- abstract wc:', len(allp[12].split()), 'conclusion wc:', len(allp[129].split()))
print('--- declarations block ---')
for i in range(130, len(allp)):
    if allp[i].strip(): print('P%d |%s|' % (i, allp[i][:150]))
