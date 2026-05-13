"""
DIAGNOSTIC SCRIPT — prints every raw NER entity from BioBERT
so we can see what entity_group labels are actually produced
for treatments, procedures, etc.
"""
import sys
sys.path.insert(0, '.')
from models.api_pipeline import pipeline

pipeline.load_models()

# Use the same clinical sample that was tested
text = """
Patient is a 71-year-old gentleman presented with shortness of breath, easy fatigability, 
and dizziness. Initial blood test in the emergency room showed elevated BNP suggestive 
of congestive heart failure. Given history and his multiple risk factors, he was admitted 
for further evaluation. His x-ray showed cardiomegaly and pneumonia.
He has a history of hypertension, hyperlipidemia, and coronary artery disease.
He is on Coumadin, Isosorbide Mononitrate, Potassium, Gemfibrozil, and Adenosine.
Doctor recommended: Continue current medications, restrict sodium intake, follow up 
echocardiogram, pulmonary function test, cardiology consultation, and bed rest.
Patient was prescribed Furosemide 40mg daily and started on oxygen therapy.
"""

print("\n" + "="*70)
print("RAW BioBERT NER OUTPUT (ALL ENTITY GROUPS)")
print("="*70)

raw = pipeline._run_ner_chunked(text)
groups = {}
for ent in raw:
    g = ent.get('entity_group', 'UNKNOWN')
    if g not in groups:
        groups[g] = []
    groups[g].append((ent['word'], round(ent['score']*100,1)))

for group, items in sorted(groups.items()):
    print(f"\n[{group}]")
    for word, score in items:
        print(f"  {score:5.1f}%  '{word}'")

print("\n" + "="*70)
print("TOTAL GROUPS FOUND:", list(groups.keys()))
