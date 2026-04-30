"""
=============================================================
MEDIREPORTER — Full Pipeline Script
=============================================================
Runs all 3 models on a medical report:
  1. LSTM  → Rough summary (loaded from saved checkpoint)
  2. BART  → Clean summary (pre-trained, no fine-tuning needed)
  3. NER   → Structured medical entities

Usage:
  Local:  python full_pipeline.py
  Colab:  Upload lstm_summarizer.pt, then run each section
=============================================================
"""

import os
import re
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ── Device ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {DEVICE}")

# =============================================================
#  SECTION 1: LSTM MODEL (Load & Infer — NO TRAINING)
# =============================================================

# ── Special tokens ──
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

MAX_ARTICLE_LEN = 400
MAX_SUMMARY_LEN = 80

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    return text.strip()

def text_to_tensor(text, vocab, max_len):
    words = text.split()[:max_len]
    ids = [vocab.get(word, vocab[UNK_TOKEN]) for word in words]
    ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# ── Model Architecture (must match training) ──
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, tgt_word, hidden, cell, encoder_outputs):
        tgt_word = tgt_word.unsqueeze(1)
        embedded = self.dropout(self.embedding(tgt_word))
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab   = vocab
        self.device  = device
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len    = tgt.shape[1]
        vocab_size = len(self.vocab)
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        dec_input = torch.full((batch_size,), self.vocab[SOS_TOKEN],
                               dtype=torch.long).to(self.device)
        for t in range(tgt_len):
            pred, hidden, cell = self.decoder(dec_input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = pred
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher else pred.argmax(1)
        return outputs


def load_lstm_model(checkpoint_path):
    """Load trained LSTM from checkpoint"""
    print("\n" + "="*60)
    print("  LOADING LSTM MODEL")
    print("="*60)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    vocab    = checkpoint['vocab']
    idx2word = checkpoint['idx2word']

    VOCAB_SIZE = len(vocab)
    EMBED_DIM  = 128
    HIDDEN_DIM = 256
    N_LAYERS   = 2
    DROPOUT    = 0.0   # No dropout during inference

    encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    model   = Seq2Seq(encoder, decoder, vocab, DEVICE).to(DEVICE)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Model loaded from: {checkpoint_path}")
    return model, vocab, idx2word


def lstm_generate_summary(model, article, vocab, idx2word, max_len=50):
    """
    Attention-based extractive summary from LSTM.
    
    The decoder hasn't converged enough for fluent generation,
    but the ATTENTION mechanism correctly identifies important
    words in the input. We use attention weights to extract
    key phrases, producing a rough but meaningful summary.
    """
    model.eval()
    unk_idx = vocab[UNK_TOKEN]
    pad_idx = vocab[PAD_TOKEN]
    sos_idx = vocab[SOS_TOKEN]

    with torch.no_grad():
        cleaned = clean_text(article)
        input_words = cleaned.split()[:MAX_ARTICLE_LEN]
        src = text_to_tensor(cleaned, vocab, MAX_ARTICLE_LEN)
        src = src.unsqueeze(0).to(DEVICE)

        encoder_outputs, hidden, cell = model.encoder(src)
        dec_input = torch.tensor([sos_idx]).to(DEVICE)

        # Accumulate attention weights across decoder steps
        attn_sum = torch.zeros(MAX_ARTICLE_LEN)
        for step in range(20):
            pred, hidden, cell = model.decoder(
                dec_input, hidden, cell, encoder_outputs
            )
            # Get attention weights from decoder
            attn_weights = model.decoder.attention(hidden[-1], encoder_outputs)
            attn_sum += attn_weights.squeeze(0).cpu()
            pred[0][unk_idx] = float('-inf')
            pred[0][pad_idx] = float('-inf')
            dec_input = pred.argmax(1)

        # Use attention to find most important input positions
        n_input = len(input_words)
        attn_scores = attn_sum[:n_input]

        # Select top-attended words, keep original order
        n_select = min(20, n_input)
        top_indices = attn_scores.argsort(descending=True)[:n_select]
        top_indices = sorted(top_indices.tolist())

        # Build extractive summary from high-attention words
        key_words = []
        for i in top_indices:
            if i < n_input:
                w = input_words[i]
                # Skip filler/stop words
                if w not in ('the','a','an','is','was','were','are','of',
                             'to','and','in','for','with','on','at','by',
                             'that','this','it','be','has','had','have'):
                    key_words.append(w)

        summary = ' '.join(key_words)
        # Clean up punctuation
        summary = re.sub(r'\s+([.,])', r'\1', summary)
        summary = re.sub(r'\s+', ' ', summary).strip()

    return summary


# =============================================================
#  SECTION 2: BART MODEL (Pre-trained — NO FINE-TUNING needed)
# =============================================================

def load_bart_model():
    """
    Load facebook/bart-large-cnn — already fine-tuned on CNN/DailyMail.
    No additional training needed. Just load and use.
    """
    from transformers import BartTokenizer, BartForConditionalGeneration

    print("\n" + "="*60)
    print("  LOADING BART MODEL (pre-trained on CNN/DailyMail)")
    print("="*60)

    MODEL_NAME = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()

    print(f"  Model: {MODEL_NAME}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def bart_generate_summary(text, model, tokenizer, max_length=150):
    """Generate summary using BART"""
    model.eval()
    inputs = tokenizer(
        text, max_length=512, truncation=True, return_tensors='pt'
    ).to(DEVICE)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# =============================================================
#  SECTION 3: BIOMEDICAL NER (Pre-trained — ZERO training)
# =============================================================

def load_ner_model():
    """
    Load biomedical NER model — pre-trained on medical literature.
    Extracts Disease, Drug, Symptom, Treatment from text.
    """
    from transformers import pipeline

    print("\n" + "="*60)
    print("  LOADING BIOMEDICAL NER MODEL")
    print("="*60)

    ner_pipeline = pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    print("  Model: d4data/biomedical-ner-all")
    print("  Ready for entity extraction")
    return ner_pipeline


def extract_medical_entities(text, ner_pipeline):
    """Extract and categorize medical entities"""
    raw_entities = ner_pipeline(text)

    # Category mapping based on model labels
    category_map = {
        'Disease_disorder':    'Disease',
        'Sign_symptom':        'Symptom',
        'Medication':          'Drug',
        'Therapeutic_or_preventive_procedure': 'Treatment',
        'Diagnostic_procedure': 'Treatment',
        'Lab_value':           'Lab Value',
        'Clinical_event':      'Clinical Event',
    }

    results = {
        'Disease':   [],
        'Drug':      [],
        'Symptom':   [],
        'Treatment': [],
    }

    for ent in raw_entities:
        label = ent['entity_group']
        word  = ent['word'].strip()
        score = ent['score']

        category = category_map.get(label, None)
        if category and category in results:
            if word not in results[category] and score > 0.5:
                results[category].append(word)

    return results, raw_entities


# =============================================================
#  SECTION 4: ROUGE EVALUATION
# =============================================================

def compute_rouge(reference, hypothesis):
    """Compute ROUGE scores between two texts"""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-2': scores['rouge2'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure,
    }


# =============================================================
#  MAIN PIPELINE
# =============================================================

def run_full_pipeline():
    """Run the complete Medical Report Summarizer pipeline"""

    # ── Medical Report Input ──
    medical_report = """
    The patient is a 56 year old male who presented to the 
    emergency department with complaints of chest pain, 
    shortness of breath, and dizziness for the past 3 days. 
    Physical examination revealed elevated blood pressure of 
    160/100 mmHg and irregular heartbeat. Laboratory tests 
    confirmed elevated blood sugar levels. The patient was 
    diagnosed with Type 2 Diabetes Mellitus and hypertension. 
    Treatment plan includes metformin 500mg twice daily, 
    lisinopril 10mg once daily, and dietary modifications. 
    Patient was advised to follow up in 2 weeks.
    """

    reference_summary = (
        "56-year-old male diagnosed with Type 2 Diabetes and hypertension. "
        "Prescribed metformin and lisinopril. Follow up in 2 weeks."
    )

    print("\n" + "#"*60)
    print("#  MEDIREPORTER — FULL PIPELINE")
    print("#"*60)
    print(f"\nINPUT REPORT ({len(medical_report.split())} words):")
    print(medical_report.strip())

    # ────────────────────────────────────────────────
    #  STEP 1: LSTM Summary
    # ────────────────────────────────────────────────
    lstm_summary = "[LSTM model not found — skipped]"
    lstm_model_path = os.path.join(os.path.dirname(__file__), "lstm_summarizer.pt")

    if os.path.exists(lstm_model_path):
        try:
            lstm_model, vocab, idx2word = load_lstm_model(lstm_model_path)
            lstm_summary = lstm_generate_summary(
                lstm_model, medical_report, vocab, idx2word
            )
            del lstm_model  # free memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            lstm_summary = f"[LSTM error: {e}]"
    else:
        print(f"\n⚠ LSTM model not found at: {lstm_model_path}")
        print("  Upload lstm_summarizer.pt to this directory.")

    # ────────────────────────────────────────────────
    #  STEP 2: BART Summary
    # ────────────────────────────────────────────────
    bart_model, bart_tokenizer = load_bart_model()
    bart_summary = bart_generate_summary(
        medical_report, bart_model, bart_tokenizer
    )
    del bart_model  # free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ────────────────────────────────────────────────
    #  STEP 3: NER Entity Extraction
    # ────────────────────────────────────────────────
    ner_pipeline = load_ner_model()
    # Run NER on the original medical report (more entities to find)
    entities, raw_entities = extract_medical_entities(medical_report, ner_pipeline)

    # ────────────────────────────────────────────────
    #  STEP 4: ROUGE Evaluation
    # ────────────────────────────────────────────────
    rouge_bart = compute_rouge(reference_summary, bart_summary)
    rouge_lstm = compute_rouge(reference_summary, lstm_summary)

    # ════════════════════════════════════════════════
    #  DISPLAY ALL RESULTS
    # ════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)

    print("\n--- LSTM SUMMARY (rough, basic output) ---")
    print(f'"{lstm_summary}"')

    print("\n--- BART SUMMARY (clean, fluent output) ---")
    print(f'"{bart_summary}"')

    print("\n--- NER ENTITIES (structured extraction) ---")
    for category, items in entities.items():
        items_str = items if items else ["(none detected)"]
        print(f"  {category:10s} -> {items_str}")

    print("\n--- ROUGE SCORES ---")
    print(f"\n  {'Metric':<10} {'BART':>10} {'LSTM':>10}")
    print(f"  {'-'*30}")
    for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
        print(f"  {metric:<10} {rouge_bart[metric]:>10.4f} {rouge_lstm[metric]:>10.4f}")

    print("\n--- RAW NER LABELS (for reference) ---")
    for ent in raw_entities:
        print(f"  [{ent['entity_group']:>40s}]  {ent['word']:<25s}  (conf: {ent['score']:.3f})")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_full_pipeline()
