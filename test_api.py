import requests
import json

text = "Patient is a 56-year-old male with chest pain and shortness of breath. Diagnosed with Type 2 Diabetes and hypertension. Prescribed metformin 500mg and lisinopril 10mg. Patient has COPD."
res = requests.post("http://localhost:8000/api/analyze", data={"text": text})
print(json.dumps(res.json(), indent=2))
