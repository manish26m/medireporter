"""
=============================================================
MEDIREPORTER — Industrial-Grade API Backend Pipeline v2.0
=============================================================
Three-stage pipeline:
  Stage 1: LSTM Attention-based Keyword Extraction (DL baseline)
  Stage 2: BART Abstractive Summarization (Transformer NLP)
  Stage 3: BioBERT Named Entity Recognition w/ noise elimination

NER Fix: Multi-layer post-processing eliminates subword noise,
low-confidence predictions, acronyms, and medical stopwords.
=============================================================
"""

import os
import re
import time
import logging
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("medireporter.pipeline")

# ── Device ────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================
#  HIGH-RISK DISEASE KEYWORDS — used for risk classification
# =============================================================
HIGH_RISK_KEYWORDS = {
    'cancer','carcinoma','tumor','tumour','malignant','malignancy',
    'myocardial infarction','heart attack','stroke','sepsis','seizure',
    'hemorrhage','haemorrhage','embolism','thrombosis','aneurysm',
    'respiratory failure','cardiac arrest','renal failure','liver failure',
    'acute', 'critical','icu','intensive care','coma','unconscious',
    'metastasis','metastatic','pulmonary edema','shock'
}

MODERATE_RISK_KEYWORDS = {
    'hypertension','diabetes','diabetic','pneumonia','infection',
    'fracture','surgery','surgical','chronic','disorder','syndrome',
    'arrhythmia','fibrillation','asthma','copd','renal','hepatic',
    'hyperlipidemia','obesity','depression','anxiety','hypo','hyper'
}

# =============================================================
#  MEDICAL STOPWORDS — these are NOT valid clinical entities
# =============================================================
MEDICAL_STOPWORDS = {
    # Common abbreviations that slip through NER
    'the','and','for','with','this','that','they','them','when',
    'what','where','have','been','from','were','will','would',
    'could','should','which','there','their','about','after',
    'before','during','through','between','patient','patients',
    'history','past','present','year','years','month','months',
    'day','days','week','weeks','male','female','aged','age',
    'time','date','report','hospital','clinic','doctor','nurse',
    'examination','exam','test','tests','result','results',
    'normal','negative','positive','noted','seen','found',
    'showed','shows','showing','noted','complaints','complaint',
    'pain','mild','moderate','severe','significant','associated',
    'bilateral','right','left','upper','lower','anterior','posterior',
    # 2-3 letter acronyms that are pure noise
    'pt','hx','cc','bp','hr','rr','wbc','rbc','hgb','hct',
    'ekg','ecg','mri','cxr','abi','ast','alt','bun','cr',
    'na','mg','kg','dl','ml','mm','cm','iv','im','po','prn',
    # Single characters or fragments
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
}

# =============================================================
#  SECTION 1: LSTM ARCHITECTURE (Deep Learning Baseline)
# =============================================================

PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
MAX_ARTICLE_LEN = 400

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    return text.strip()

def text_to_tensor(text: str, vocab: dict, max_len: int) -> torch.Tensor:
    words = text.split()[:max_len]
    ids = [vocab.get(word, vocab[UNK_TOKEN]) for word in words]
    ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


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
        hidden  = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy  = torch.tanh(self.attn(
            torch.cat((hidden, encoder_outputs), dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm      = nn.LSTM(embed_dim + hidden_dim, hidden_dim,
                                 n_layers, batch_first=True, dropout=dropout)
        self.fc_out    = nn.Linear(hidden_dim, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, tgt_word, hidden, cell, encoder_outputs):
        tgt_word     = tgt_word.unsqueeze(1)
        embedded     = self.dropout(self.embedding(tgt_word))
        attn_weights = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        context      = torch.bmm(attn_weights, encoder_outputs)
        lstm_input   = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction   = self.fc_out(output.squeeze(1))
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
        outputs    = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        enc_out, hidden, cell = self.encoder(src)
        dec_input = torch.full(
            (batch_size,), self.vocab[SOS_TOKEN], dtype=torch.long).to(self.device)
        for t in range(tgt_len):
            pred, hidden, cell = self.decoder(dec_input, hidden, cell, enc_out)
            outputs[:, t, :] = pred
            use_tf   = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_tf else pred.argmax(1)
        return outputs


def lstm_generate_summary(model, article, vocab, idx2word, max_len=50) -> str:
    """
    Attention-based keyword extraction using the LSTM encoder.
    Returns top-20 high-attention content words in source order.
    """
    model.eval()
    unk_idx = vocab[UNK_TOKEN]
    pad_idx = vocab[PAD_TOKEN]
    sos_idx = vocab[SOS_TOKEN]

    with torch.no_grad():
        cleaned     = clean_text(article)
        input_words = cleaned.split()[:MAX_ARTICLE_LEN]
        src = text_to_tensor(cleaned, vocab, MAX_ARTICLE_LEN).unsqueeze(0).to(DEVICE)

        enc_out, hidden, cell = model.encoder(src)
        dec_input = torch.tensor([sos_idx]).to(DEVICE)

        attn_sum = torch.zeros(MAX_ARTICLE_LEN)
        for _ in range(20):
            pred, hidden, cell = model.decoder(dec_input, hidden, cell, enc_out)
            attn_w = model.decoder.attention(hidden[-1], enc_out)
            attn_sum += attn_w.squeeze(0).cpu()
            pred[0][unk_idx] = float('-inf')
            pred[0][pad_idx] = float('-inf')
            dec_input = pred.argmax(1)

        n_input     = len(input_words)
        attn_scores = attn_sum[:n_input]
        n_select    = min(20, n_input)
        top_indices = sorted(attn_scores.argsort(descending=True)[:n_select].tolist())

        # Collect significant content words
        content_stopwords = {
            'the','a','an','is','was','were','are','of','to','and',
            'in','for','with','on','at','by','that','this','it',
            'be','has','had','have','been','will','would','could'
        }
        key_words = [
            input_words[i] for i in top_indices
            if i < n_input and input_words[i] not in content_stopwords
        ]

        summary = ' '.join(key_words)
        summary = re.sub(r'\s+([.,])', r'\1', summary)
        return re.sub(r'\s+', ' ', summary).strip()


# =============================================================
#  SECTION 2: NER POST-PROCESSING (Noise Elimination)
# =============================================================

def is_valid_entity(word: str, score: float,
                    score_threshold: float = 0.87) -> bool:
    """
    Multi-layer validation to eliminate NER noise from BioBERT.

    Rejects:
    - Subword fragments starting with ## or containing ##
    - Tokens shorter than 4 characters
    - Pure numeric strings
    - Tokens with special characters (except hyphens)
    - Tokens in the medical stopword list
    - Tokens below confidence threshold
    """
    # 1. Remove leading/trailing whitespace
    word = word.strip()

    # 2. Must have sufficient length
    if len(word) < 4:
        return False

    # 3. Reject subword artifacts
    if '##' in word:
        return False

    # 4. Reject pure numbers
    if re.fullmatch(r'[\d\s.,]+', word):
        return False

    # 5. Reject tokens with illegal special characters
    if re.search(r'[^a-zA-Z0-9\-\s]', word):
        return False

    # 6. Reject known medical stopwords (case-insensitive)
    if word.lower() in MEDICAL_STOPWORDS:
        return False

    # 7. Reject tokens that are entirely uppercase acronyms <= 4 chars
    if word.isupper() and len(word) <= 4:
        return False

    # 8. Confidence threshold check
    if score < score_threshold:
        return False

    return True


def merge_subword_entities(raw_entities: list) -> list:
    """
    Robustly merge BERT subword tokens back into full words.
    Handles both ## prefix subwords and space-separated multi-tokens.
    """
    merged = []
    for ent in raw_entities:
        word  = ent['word']
        # Case 1: Standard ## subword continuation
        if word.startswith('##') and merged:
            merged[-1]['word'] += word[2:]
            merged[-1]['end']   = ent['end']
            # Keep the minimum score (weakest link)
            merged[-1]['score'] = min(merged[-1]['score'], ent['score'])
        # Case 2: Space-separated token that continues the previous span
        elif merged and ent.get('start', -1) == merged[-1].get('end', -2):
            merged[-1]['word'] += word
            merged[-1]['end']   = ent['end']
            merged[-1]['score'] = min(merged[-1]['score'], ent['score'])
        else:
            merged.append(dict(ent))
    return merged


def deduplicate_entities(entities: list) -> list:
    """Remove duplicates case-insensitively, keeping highest-scored version."""
    seen   = {}
    result = []
    for ent in entities:
        key = ent.lower()
        if key not in seen:
            seen[key] = True
            result.append(ent)
    return result


def classify_risk(entities: dict, text: str) -> dict:
    """
    Rule-based risk classification using entity counts + keyword matching.
    Returns { level: 'Low'|'Moderate'|'High', score: 0-100, reason: str }
    """
    text_lower  = text.lower()
    all_entities = (
        entities.get('Disease', []) +
        entities.get('Symptom', []) +
        entities.get('Drug', []) +
        entities.get('Treatment', [])
    )

    high_hits = [kw for kw in HIGH_RISK_KEYWORDS
                 if kw in text_lower]
    mod_hits  = [kw for kw in MODERATE_RISK_KEYWORDS
                 if kw in text_lower]

    entity_count = len(all_entities)
    disease_count = len(entities.get('Disease', []))

    # Score: 0-100
    score = 0
    score += len(high_hits) * 25
    score += len(mod_hits)  * 12
    score += min(disease_count * 8, 40)
    score += min(entity_count  * 2, 20)
    score = min(score, 100)

    if score >= 60 or len(high_hits) >= 1:
        level  = 'High'
        reason = f"High-risk indicators detected: {', '.join(high_hits[:3]) or 'multiple conditions'}"
    elif score >= 30 or len(mod_hits) >= 2:
        level  = 'Moderate'
        reason = f"Moderate conditions: {', '.join(mod_hits[:3]) or 'multiple comorbidities'}"
    else:
        level  = 'Low'
        reason = "No critical high-risk indicators detected."

    return {'level': level, 'score': score, 'reason': reason}


def calculate_confidence(entities: dict, bart_summary: str, lstm_summary: str) -> dict:
    """
    Compute overall pipeline confidence as a composite metric.
    """
    total_entities = sum(len(v) for v in entities.values())
    entity_conf    = min(total_entities / 10.0, 1.0)  # normalized 0-1

    bart_conf = min(len(bart_summary.split()) / 30.0, 1.0) if bart_summary else 0.0
    lstm_conf = min(len(lstm_summary.split()) / 10.0, 1.0) if lstm_summary else 0.0

    overall = round((entity_conf * 0.5 + bart_conf * 0.4 + lstm_conf * 0.1) * 100, 1)
    return {
        'overall_pct': overall,
        'entity_confidence': round(entity_conf * 100, 1),
        'bart_confidence':   round(bart_conf   * 100, 1),
        'lstm_confidence':   round(lstm_conf   * 100, 1),
    }


# =============================================================
#  SECTION 3: PIPELINE STATE
# =============================================================

class MediPipeline:
    VERSION = "2.0.0"

    def __init__(self):
        self.lstm_model      = None
        self.vocab           = None
        self.idx2word        = None
        self.bart_model      = None
        self.bart_tokenizer  = None
        self.ner_pipeline    = None
        self._models_loaded  = False

    # ----------------------------------------------------------
    def load_models(self):
        t0 = time.time()
        logger.info("=== MediPipeline Model Loading Started ===")

        # ── LSTM ──────────────────────────────────────────────
        lstm_path = os.path.join(os.path.dirname(__file__), "lstm_summarizer.pt")
        if os.path.exists(lstm_path):
            logger.info("Loading LSTM model from %s", lstm_path)
            cp = torch.load(lstm_path, map_location=DEVICE, weights_only=False)
            self.vocab    = cp['vocab']
            self.idx2word = cp['idx2word']
            VS  = len(self.vocab)
            enc = Encoder(VS, 128, 256, 2, 0.0)
            dec = Decoder(VS, 128, 256, 2, 0.0)
            self.lstm_model = Seq2Seq(enc, dec, self.vocab, DEVICE).to(DEVICE)
            self.lstm_model.load_state_dict(cp['model_state'])
            self.lstm_model.eval()
            logger.info("LSTM model loaded  (vocab=%d tokens)", VS)
        else:
            logger.warning("lstm_summarizer.pt not found — LSTM stage disabled.")

        # ── BART ──────────────────────────────────────────────
        logger.info("Loading BART model (facebook/bart-large-cnn)…")
        from transformers import BartTokenizer, BartForConditionalGeneration
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large-cnn").to(DEVICE)
        self.bart_model.eval()
        logger.info("BART model loaded.")

        # ── BioBERT NER ───────────────────────────────────────
        logger.info("Loading BioBERT NER model (d4data/biomedical-ner-all)…")
        from transformers import pipeline as hf_pipeline
        self.ner_pipeline = hf_pipeline(
            "ner",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("BioBERT NER loaded.")

        self._models_loaded = True
        elapsed = round(time.time() - t0, 1)
        logger.info("=== All models loaded in %.1fs ===", elapsed)

    # ----------------------------------------------------------
    def _run_ner_chunked(self, text: str) -> list:
        """
        Run NER on text in 512-char chunks to avoid BERT token limit.
        Aggregates results from all chunks.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks    = []
        current   = ""
        for sent in sentences:
            if len(current) + len(sent) < 450:
                current += " " + sent
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent
        if current.strip():
            chunks.append(current.strip())

        all_raw = []
        for chunk in chunks:
            try:
                all_raw.extend(self.ner_pipeline(chunk))
            except Exception as e:
                logger.warning("NER chunk failed: %s", str(e))
        return all_raw

    # ----------------------------------------------------------
    def process(self, text: str) -> dict:
        if not self._models_loaded:
            raise RuntimeError("Models are not loaded yet. Call load_models() first.")

        t_start = time.time()
        word_count = len(text.split())

        # ── Stage 1: LSTM Keyword Extraction ──────────────────
        lstm_out = "LSTM model not available on this deployment."
        if self.lstm_model:
            try:
                lstm_out = lstm_generate_summary(
                    self.lstm_model, text, self.vocab, self.idx2word)
                logger.debug("LSTM summary: %s", lstm_out[:80])
            except Exception as e:
                lstm_out = f"LSTM error: {str(e)}"
                logger.error("LSTM failed: %s", str(e))

        # ── Stage 2: BART Abstractive Summarization ───────────
        bart_out = ""
        try:
            inputs = self.bart_tokenizer(
                text, max_length=1024, truncation=True, return_tensors='pt'
            ).to(DEVICE)
            with torch.no_grad():
                summary_ids = self.bart_model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=180,
                    min_length=40,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            bart_out = self.bart_tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            logger.debug("BART summary: %s", bart_out[:80])
        except Exception as e:
            bart_out = f"BART error: {str(e)}"
            logger.error("BART failed: %s", str(e))

        # ── Stage 3: BioBERT NER with Noise Elimination ───────
        CAT_MAP = {
            'Disease_disorder':                    'Disease',
            'Sign_symptom':                        'Symptom',
            'Medication':                          'Drug',
            'Therapeutic_or_preventive_procedure': 'Treatment',
            'Diagnostic_procedure':                'Treatment',
            'Biological_structure':                'Treatment',
        }
        entities: dict[str, list] = {
            'Disease': [], 'Drug': [], 'Symptom': [], 'Treatment': []
        }
        entity_scores: dict[str, list] = {
            'Disease': [], 'Drug': [], 'Symptom': [], 'Treatment': []
        }

        try:
            raw_entities  = self._run_ner_chunked(text)
            merged        = merge_subword_entities(raw_entities)

            for ent in merged:
                cat = CAT_MAP.get(ent.get('entity_group', ''))
                if not cat:
                    continue
                word  = ent['word'].strip()
                score = float(ent.get('score', 0.0))

                # Apply multi-layer noise filter
                if not is_valid_entity(word, score, score_threshold=0.87):
                    continue

                # Title-case the entity for presentation
                display_word = word.title() if not any(c.isupper() for c in word[1:]) else word

                if display_word not in entities[cat]:
                    entities[cat].append(display_word)
                    entity_scores[cat].append(round(score * 100, 1))

            # Deduplicate within each category
            for cat in entities:
                entities[cat] = deduplicate_entities(entities[cat])

        except Exception as e:
            logger.error("NER failed: %s", str(e))

        # ── Stage 4: Risk Classification & Confidence ─────────
        risk       = classify_risk(entities, text)
        confidence = calculate_confidence(entities, bart_out, lstm_out)

        elapsed = round(time.time() - t_start, 2)

        return {
            "lstm_summary":   lstm_out,
            "bart_summary":   bart_out,
            "entities":       entities,
            "entity_scores":  entity_scores,
            "risk":           risk,
            "confidence":     confidence,
            "metadata": {
                "pipeline_version": self.VERSION,
                "device":           str(DEVICE),
                "word_count":       word_count,
                "processing_time_s": elapsed,
                "lstm_available":   self.lstm_model is not None,
                "models": {
                    "summarizer": "facebook/bart-large-cnn",
                    "ner":        "d4data/biomedical-ner-all",
                    "baseline":   "LSTM Seq2Seq + Bahdanau Attention"
                }
            }
        }

    def get_status(self) -> dict:
        return {
            "models_loaded": self._models_loaded,
            "lstm_available": self.lstm_model is not None,
            "bart_available": self.bart_model is not None,
            "ner_available":  self.ner_pipeline is not None,
            "device":         str(DEVICE),
            "version":        self.VERSION,
        }


# Global singleton
pipeline = MediPipeline()
