import json
from models.api_pipeline import pipeline

pipeline.load_models()
text = "Patient is a 56-year-old male with chest pain and shortness of breath. Diagnosed with Type 2 Diabetes and hypertension. Prescribed metformin 500mg and lisinopril 10mg. Patient has COPD."
raw = pipeline.ner_pipeline(text)
print("RAW NER:")
for r in raw: print(r)

