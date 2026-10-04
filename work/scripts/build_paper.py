"""Build the public research paper from capstone markdown and its metric receipts."""
import json
from pathlib import Path
import re
import shutil
import markdown

ROOT = Path(__file__).resolve().parents[2]
nb = json.loads((ROOT / 'work/notebooks/capstone.ipynb').read_text(encoding='utf-8'))
sections = []
for cell in nb['cells']:
    if cell['cell_type'] != 'markdown':
        continue
    source = ''.join(cell['source'])
    if source.startswith('## 9.'):
        break
    sections.append(source)
paper = '\n\n'.join(sections)
receipt = json.loads((ROOT / 'work/outputs/w07_playbook_metrics.json').read_text())
for method, label in [('Frozen rule / all OOF pages', 'Frozen rule'),
                      ('Logistic regression / all OOF pages', 'Logistic regression')]:
    metrics = receipt['validated_metrics'][method]
    line = (f'| {label} | {100*metrics["macro_precision_at_20"]:.2f}% | '
            f'{100*metrics["macro_base_rate_same_clients"]:.2f}% | '
            f'{metrics["roc_auc"]:.4f} | {metrics["average_precision"]:.4f} |')
    assert line in paper, f'Update the paper table to match the receipt: {label}'
assert not re.search(r'hf_[A-Za-z0-9]{20,}', paper)
assert 'https://flyrank.ai' in paper
assert all(f'## {i}.' in paper for i in range(1,9))
body = markdown.markdown(paper, extensions=['tables'])
css = '''
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f8f7f3;color:#242b2c;font:18px/1.7 Georgia,serif}
main{max-width:940px;margin:0 auto;padding:64px 28px 90px}header,footer{font:14px/1.5 system-ui,sans-serif;color:#596365}
header{border-bottom:1px solid #d8dcda;padding-bottom:20px;margin-bottom:38px}h1{font-size:clamp(34px,5vw,52px);line-height:1.16;letter-spacing:-.025em;max-width:780px}
h2{font:600 25px/1.3 system-ui,sans-serif;margin-top:52px}p{margin:1em 0}a{color:#13685f;text-underline-offset:3px}a:focus-visible{outline:3px solid #13685f;outline-offset:3px}
img{max-width:100%;height:auto;border:1px solid #d8dcda}code{font-size:.83em;overflow-wrap:anywhere}table{display:block;width:100%;overflow-x:auto;border-collapse:collapse;font:14px/1.55 system-ui,sans-serif;margin:24px 0}
th,td{text-align:left;padding:12px;border-bottom:1px solid #d8dcda;min-width:120px}th{background:#eceee9}footer{margin-top:56px;border-top:1px solid #d8dcda;padding-top:20px}
@media(max-width:600px){main{padding:28px 20px 60px}body{font-size:17px}h2{font-size:22px}}@media print{body{background:white}main{padding:0}a{color:inherit}}
'''
html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Which content pages should a FlyRank editor review first?</title>
<meta name="description" content="A FlyRank content review case study comparing a transparent rule with logistic regression under validation grouped by client. Historical research with explicit limits.">
<style>{css}</style></head><body><main>
<header>Vibhanshu · FlyRank internship research · March 2026 data<br><a href="https://github.com/vibhanshu-s/FlyrankAI-Internship">Code, executed notebooks and metric receipts</a></header>
{body}
<footer>Historical research · Human review required · No automatic content changes</footer>
</main></body></html>'''
(ROOT / 'work/capstone_report.md').write_text(paper + '\n', encoding='utf-8')
(ROOT / 'docs/index.html').write_text(html, encoding='utf-8')
(ROOT / 'docs/.nojekyll').touch()
(ROOT / 'docs/figures').mkdir(exist_ok=True)
shutil.copyfile(ROOT / 'work/figures/model_vs_baseline_precision.png',
                ROOT / 'docs/figures/model_vs_baseline_precision.png')
print('Built paper, report and served chart; metric table matches the committed receipt.')
