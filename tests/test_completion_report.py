import hashlib,json,subprocess,sys,unittest,os
from html.parser import HTMLParser
from pathlib import Path
from neuralduck.completion_report import ROOT,FEATURE
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=set();self.images=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.add(a['id'])
  if tag=='a':self.links.append(a['href'])
  if tag=='img':self.images.append(a['src'])
class ReportTests(unittest.TestCase):
 def test_regeneration_preserves_models_and_has_valid_resources(self):
  model_files=list(FEATURE.glob('*.pt'))+list(FEATURE.glob('*.joblib'));before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in model_files}
  result=subprocess.run([os.environ.get('COMSPEC','cmd.exe'),'/d','/c','Open-EEG-Research-Report.cmd','--no-open'],cwd=ROOT,capture_output=True,text=True,env={**os.environ,"NEURALDUCK_PYTHON":sys.executable})
  self.assertEqual(result.returncode,0,result.stderr)
  for p,key in before.items():self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),key)
  out=ROOT/'artifacts/consolidated-report';parser=Links();parser.feed((out/'report.html').read_text(encoding='utf-8'))
  self.assertEqual(len(parser.images),5)
  for link in parser.links+parser.images:
   if link.startswith('#'):self.assertIn(link[1:],parser.ids)
   else:self.assertTrue((out/link).is_file(),link)
  for file in ['paired.json','classification-details.json','verification.json','inference-contract.json']:json.loads((out/file).read_text())
  self.assertEqual(len(json.loads((out/'verification.json').read_text())['results']),55)
if __name__=='__main__':unittest.main()
