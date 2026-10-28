from pathlib import Path
import html,re,shutil
from markdown_it import MarkdownIt

ROOT=Path(__file__).resolve().parent
DIST=ROOT/'dist';DIST.mkdir(exist_ok=True)
md=MarkdownIt('commonmark',{'html':False}).enable('table').enable('strikethrough')
weeks=[]
for n in range(1,13):
 folder=ROOT.parent/f'week{n}'
 file=next(folder.glob('*.md'))
 text=file.read_text()
 title=text.splitlines()[0].lstrip('# ').split(':',1)[-1].strip().removesuffix(' Notes')
 weeks.append((n,title,file,text))

def page_name(n):return 'index.html' if n==1 else f'week{n}.html'

for n,title,file,text in weeks:
 maths=[]
 def protect(m):
  maths.append(m.group(0));return f'MATHPLACEHOLDER{len(maths)-1}END'
 masked=re.sub(r'\$\$[\s\S]*?\$\$|(?<!\\)\$(?!\$)(?:\\.|[^$\n])*?(?<!\\)\$',protect,text)
 tokens=md.parse(masked);toc=[]
 for i,t in enumerate(tokens):
  if t.type=='heading_open':
   label=tokens[i+1].content
   ident=f'topic-{len(toc)}';t.attrSet('id',ident)
   toc.append((t.tag,ident,label))
 body=md.renderer.render(tokens,md.options,{})
 for i,m in enumerate(maths):body=body.replace(f'MATHPLACEHOLDER{i}END',html.escape(m))
 body=re.sub(r'src="images/([^\"]+)"',lambda m:f'src="assets/week{n}/{m[1]}" loading="lazy"',body)
 for rel in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):
  source=file.parent/rel;assert source.is_file(),source
  dest=DIST/'assets'/f'week{n}'/Path(rel).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 def clean(label):
  for i,m in enumerate(maths):label=label.replace(f'MATHPLACEHOLDER{i}END',m)
  return html.escape(label.replace('**',''))
 nav=''.join(f'<a class="week-link {"selected" if k==n else ""}" href="{page_name(k)}" {"aria-current=page" if k==n else ""}><span class="week-number">{k:02}</span><span>{html.escape(t)}</span></a>' for k,t,_,_ in weeks)
 index=''.join(f'<a class="level-{level}" href="#{ident}">{clean(label)}</a>' for level,ident,label in toc if level in ['h2','h3'])
 prev=f'<a href="{page_name(n-1)}">← Week {n-1}</a>' if n>1 else '<span></span>'
 nxt=f'<a href="{page_name(n+1)}">Week {n+1} →</a>' if n<12 else '<span></span>'
 page=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Week {n} · {html.escape(title)} | MLT Notes</title>
<script src="theme.js"></script>
<link rel="stylesheet" href="style.css">
<script>window.MathJax={{tex:{{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:true}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}},svg:{{fontCache:'global'}}}};</script>
<script defer src="vendor/tex-svg.js"></script></head>
<body><a class="skip" href="#notes">Skip to notes</a>
<header><a class="brand" href="index.html"><span class="brand-mark">m</span> MLT <span class="brand-muted">/ study notes</span></a><label class="theme-control">Theme <select id="theme"><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label></header>
<div class="layout"><aside class="weeks"><details open><summary>Course weeks <span>12</span></summary><nav aria-label="Course weeks">{nav}</nav></details></aside>
<main id="notes"><div class="eyebrow">MACHINE LEARNING TECHNIQUES <span>/ WEEK {n:02}</span></div><article>{body}</article><nav class="next-prev" aria-label="Previous and next weeks">{prev}{nxt}</nav></main>
<aside class="contents"><details open><summary>On this page</summary><nav aria-label="Topic index">{index}</nav></details></aside></div>
<footer>MLT Notes <span>Learn one concept at a time.</span></footer>
</body></html>'''
 (DIST/page_name(n)).write_text(page)
print('Built 12 weeks with complete notes and topic indexes.')
