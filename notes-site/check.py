from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import re
root=Path(__file__).resolve().parent
class Page(HTMLParser):
 def __init__(self,text):
  super().__init__(); self.ids=[];self.links=[];self.feed(text)
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  for key in ('src','href'):
   if key in a:self.links.append(a[key])
count=0
for n in range(1,13):
 src=next((root.parent/f'week{n}').glob('*.md'))
 path=root/'dist'/('index.html' if n==1 else f'week{n}.html')
 text=path.read_text();page=Page(text)
 assert 'MATHPLACEHOLDER' not in text
 assert len(page.ids)==len(set(page.ids)),path
 assert len(re.findall(r'^#{1,6} ',src.read_text(),re.M))==len([i for i in page.ids if i.startswith("topic-")]),path
 for link in page.links:
  url=urlsplit(link)
  if url.scheme or url.netloc:continue
  target=path.parent/unquote(url.path) if url.path else path
  assert target.is_file(),(path,link)
  if url.fragment:
   other=page if target==path else Page(target.read_text())
   assert unquote(url.fragment) in other.ids,(path,link)
  count+=1
print(f'Passed: 12 original note sources, heading indexes, unique anchors, and {count} local links/assets.')
