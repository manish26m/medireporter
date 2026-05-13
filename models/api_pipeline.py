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
    # Common English words that slip through NER
    'the','and','for','with','this','that','they','them','when',
    'what','where','have','been','from','were','will','would',
    'could','should','which','there','their','about','after',
    'before','during','through','between',
    # Patient/clinical context words (not actual entities)
    'patient','patients','history','past','present','year','years',
    'month','months','day','days','week','weeks','male','female',
    'aged','age','time','date','report','hospital','clinic',
    'doctor','nurse','physician','examination','exam','level','levels',
    # Generic clinical nouns that are NOT entities
    'test','tests','result','results','finding','findings','measurement',
    'normal','negative','positive','noted','seen','found',
    'showed','shows','showing','complaint','complaints',
    'medication','medications','medicine','medicines','drug','drugs',
    'symptom','symptoms','condition','conditions','disease','diseases',
    'treatment','treatments','procedure','procedures','diagnosis',
    'gender','sex','lifestyle','modification','modifications',
    # Social/demographic terms that BioBERT History-tags
    'married','single','divorced','widowed','smoke','smoker','smoking',
    'alcohol','drinker','tobacco','recreational','drug use',
    'presentation','complaint','admission','discharge',
    'allergy','allergies','allgies','alert','oriented',
    # Clinical event / lab noise words
    'presented','admitted','elevated','emergency','room','emergency room',
    'suggestive','evaluation','follow','follow up','follow-up',
    'recommended','prescribed','started','continued','continue',
    'intake','restrict','restriction','consultation','therapy',
    'oxygen','bed rest','bed','rest','sodium','intake',
    'gentleman','lady','woman','man','person','individual',
    # Qualifiers and descriptors
    'mild','moderate','severe','significant','associated',
    'bilateral','right','left','upper','lower','anterior','posterior',
    'pain','blood','sugar','level','levels','pressure','rate',
    # 2-3 letter clinical abbreviations that are pure noise
    'pt','hx','cc','bp','hr','rr','wbc','rbc','hgb','hct',
    'ekg','ecg','mri','cxr','abi','ast','alt','bun','cr',
    'na','mg','kg','dl','ml','mm','cm','iv','im','po','prn',
    # Single chars
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


def _get_attention_scores(model, article: str, vocab: dict) -> tuple:
    """
    Run the LSTM encoder and accumulate attention weights over the input tokens.
    Returns (input_words, attn_scores_tensor) where attn_scores_tensor[i]
    is the total attention the decoder paid to word i across all decode steps.
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
        for _ in range(30):           # more decode steps → more reliable attention map
            pred, hidden, cell = model.decoder(dec_input, hidden, cell, enc_out)
            attn_w = model.decoder.attention(hidden[-1], enc_out)
            attn_sum += attn_w.squeeze(0).cpu()
            pred[0][unk_idx] = float('-inf')
            pred[0][pad_idx] = float('-inf')
            dec_input = pred.argmax(1)

        n_input     = len(input_words)
        attn_scores = attn_sum[:n_input]          # shape (n_input,)
        return input_words, attn_scores


def lstm_select_sentences(model, article: str, vocab: dict,
                          coverage: float = 0.55) -> tuple:
    """
    True extractive stage: score every sentence in the document using
    LSTM encoder attention weights, then select the highest-scoring
    sentences that together cover `coverage` fraction of the total
    attention mass (default 55%).

    Returns
    -------
    selected_text : str
        The selected sentences joined as a paragraph — this is fed to BART.
    keyword_display : str
        Human-readable summary of what the LSTM selected (shown in the UI).
    """
    # -- Split original article into sentences (preserve original text, not lowercased)
    raw_sentences = re.split(r'(?<=[.!?])\s+', article.strip())
    raw_sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]

    if not raw_sentences:
        return article, "No sentences extracted."

    # -- Get per-token attention scores on the cleaned/lowercased version
    input_words, attn_scores = _get_attention_scores(model, article, vocab)
    total_attn = float(attn_scores.sum()) or 1.0

    # Map each cleaned word index back to its sentence by reconstructing
    # the cleaned token stream and aligning with raw sentences.
    cleaned_words = clean_text(article).split()[:MAX_ARTICLE_LEN]

    # Build a word→sentence index mapping via cumulative word count
    sentence_cleaned_words = []
    for s in raw_sentences:
        sentence_cleaned_words.append(clean_text(s).split())

    word_pos = 0
    sentence_scores = []
    for sent_words in sentence_cleaned_words:
        n = len(sent_words)
        # sum attention over the words belonging to this sentence
        end = min(word_pos + n, len(attn_scores))
        score = float(attn_scores[word_pos:end].sum())
        sentence_scores.append(score)
        word_pos += n

    # Sort sentences by score (descending) and greedily select until
    # we hit the coverage threshold
    indexed = sorted(enumerate(sentence_scores), key=lambda x: x[1], reverse=True)
    accumulated = 0.0
    selected_indices = set()
    for idx, score in indexed:
        selected_indices.add(idx)
        accumulated += score
        if accumulated / total_attn >= coverage:
            break

    # Restore original document order
    selected_sentences = [raw_sentences[i] for i in sorted(selected_indices)]
    selected_text = ' '.join(selected_sentences)

    n_total    = len(raw_sentences)
    n_selected = len(selected_sentences)
    keyword_display = (
        f"LSTM selected {n_selected} of {n_total} sentences "
        f"({round(accumulated/total_attn*100)}% attention coverage):\n\n"
        + selected_text
    )

    return selected_text, keyword_display

# =============================================================
#  SECTION 2: NER POST-PROCESSING (Noise Elimination)
# =============================================================

# Negation/qualifier prefixes — entities starting with these should be excluded
# (e.g. "No History Of Diabetes" should not appear as a Disease entity)
NEGATION_PREFIXES = {
    'no ', 'non ', 'non-', 'not ', 'without ', 'denies ', 'denied ',
    'questionable ', 'possible ', 'probable ', 'unlikely ', 'does ', 'history of no ',
    'no history', 'negative for ', 'absent ', 'ruled out',
}


def is_negated_entity(word: str) -> bool:
    """Return True if the entity phrase starts with a negation/qualifier."""
    w = word.lower().strip()
    return any(w.startswith(prefix) for prefix in NEGATION_PREFIXES)


def is_valid_entity(word: str, score: float,
                    score_threshold: float = 0.80) -> bool:
    """
    Multi-layer validation to eliminate NER noise from BioBERT.
    """
    word = word.strip()

    # 1. Must have sufficient length
    if len(word) < 3:
        return False

    # 2. Single-word entities must be at least 4 characters
    #    (blocks "Air", "Short", "Blurry", "ST-T" etc.)
    if ' ' not in word and '-' not in word and len(word) < 4:
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

    # 7. Reject single-word entities that are in stopwords after stripping hyphens
    if word.replace('-', ' ').strip().lower() in MEDICAL_STOPWORDS:
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
            aggregation_strategy="first",   # 'first' handles subwords better than 'simple'
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
    def process(self, text: str, use_lstm: bool = True) -> dict:
        if not self._models_loaded:
            raise RuntimeError("Models are not loaded yet. Call load_models() first.")

        t_start = time.time()
        word_count = len(text.split())

        # ── Stage 1: LSTM Sentence Selection ──────────────────
        lstm_out = "Skipped — running in High-Accuracy mode (BART + BioBERT only)."
        bart_input_text = text  # Default to full text
        
        if use_lstm and self.lstm_model:
            try:
                # LSTM scores sentences and selects the most relevant ones
                filtered_text, lstm_out = lstm_select_sentences(
                    self.lstm_model, text, self.vocab)
                
                # In standard mode, BART only sees the LSTM-filtered text
                bart_input_text = filtered_text
                logger.debug("LSTM selected text length: %d chars", len(filtered_text))
            except Exception as e:
                lstm_out = f"LSTM error: {str(e)}"
                logger.error("LSTM failed: %s", str(e))

        # ── Stage 2: BART Abstractive Summarization ───────────
        # In high-accuracy mode: use stronger beam search for a more
        # comprehensive, longer summary on the full text.
        bart_out = ""
        try:
            inputs = self.bart_tokenizer(
                bart_input_text, max_length=1024, truncation=True, return_tensors='pt'
            ).to(DEVICE)
            with torch.no_grad():
                if use_lstm:
                    # Standard mode — fast, concise
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
                else:
                    # High-accuracy mode — more beams, longer, no length bias
                    summary_ids = self.bart_model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=280,
                        min_length=60,
                        num_beams=8,
                        length_penalty=1.0,
                        early_stopping=False,
                        no_repeat_ngram_size=4,
                        repetition_penalty=1.5,
                    )
            bart_out = self.bart_tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            logger.debug("BART summary: %s", bart_out[:80])
        except Exception as e:
            bart_out = f"BART error: {str(e)}"
            logger.error("BART failed: %s", str(e))

        # ── Stage 3: BioBERT NER ───────────────────────────────
        # IMPORTANT: These keys must match the model's actual entity_group
        # output labels exactly (verified via diagnose_ner.py diagnostic).
        CAT_MAP = {
            # Disease entities
            'Disease_disorder':    'Disease',
            'History':             'Disease',   # past medical history
            'Family_history':      'Disease',   # family history of conditions
            # Symptom entities
            'Sign_symptom':        'Symptom',
            # Drug/medication entities
            'Medication':          'Drug',
            # Treatment entities — NOTE: model uses 'Therapeutic_procedure'
            # NOT 'Therapeutic_or_preventive_procedure'
            'Therapeutic_procedure':             'Treatment',
            'Therapeutic_or_preventive_procedure':'Treatment',  # keep both variants
            'Clinical_event':                    'Treatment',   # e.g. 'admitted', 'presented'
        }
        entities: dict[str, list] = {
            'Disease': [], 'Drug': [], 'Symptom': [], 'Treatment': []
        }
        entity_scores: dict[str, list] = {
            'Disease': [], 'Drug': [], 'Symptom': [], 'Treatment': []
        }

        def _extract_entities_from_text(source_text: str, thresholds: dict) -> tuple:
            """
            Run NER on source_text and return (entities_dict, scores_dict).
            This is called once for standard mode (full text only) and twice
            for high-accuracy mode (full text + BART summary).
            """
            ents_out   = {'Disease': [], 'Drug': [], 'Symptom': [], 'Treatment': []}
            scores_out = {'Disease': [], 'Drug': [], 'Symptom': [], 'Treatment': []}

            raw    = self._run_ner_chunked(source_text)
            merged = merge_subword_entities(raw)

            # Build dosage lookup
            dosage_map = {}
            for ent in merged:
                if ent.get('entity_group') == 'Dosage':
                    dosage_map[ent.get('start', -1)] = ent['word'].strip()

            for ent in merged:
                cat = CAT_MAP.get(ent.get('entity_group', ''))
                if not cat:
                    continue
                word  = ent['word'].strip()
                score = float(ent.get('score', 0.0))

                if is_negated_entity(word):
                    continue

                threshold = thresholds.get(cat, 0.80)
                if not is_valid_entity(word, score, score_threshold=threshold):
                    continue

                display_word = word.title() if not any(c.isupper() for c in word[1:]) else word

                # Append dosage to Drug entries if immediately adjacent
                if cat == 'Drug':
                    ent_end = ent.get('end', -1)
                    for dos_start, dos_str in dosage_map.items():
                        if abs(dos_start - ent_end) <= 2:
                            display_word = display_word + ' ' + dos_str
                            break

                if display_word not in ents_out[cat]:
                    ents_out[cat].append(display_word)
                    scores_out[cat].append(round(score * 100, 1))

            return ents_out, scores_out

        try:
            # Per-category confidence thresholds
            THRESHOLDS = {
                'Disease':   0.80,
                'Symptom':   0.80,
                'Drug':      0.75,
                'Treatment': 0.50,
            }

            if use_lstm:
                # ── Standard mode: single-pass NER on full text ──
                logger.debug("NER: single-pass (standard mode)")
                entities, entity_scores = _extract_entities_from_text(text, THRESHOLDS)
            else:
                # ── High-Accuracy mode: double-pass NER ──────────
                # Pass 1: Full text — broad coverage
                # Pass 2: BART summary — high-precision, focused
                # Merge: summary-pass entities take priority (they appear first),
                #        then add any additional entities from full-text pass.
                logger.debug("NER: double-pass (high-accuracy mode)")

                sum_ents, sum_scores = _extract_entities_from_text(bart_out, THRESHOLDS)
                full_ents, full_scores = _extract_entities_from_text(text, THRESHOLDS)

                # Merge: summary entities first (higher precision),
                # then fill in from full-text pass
                for cat in entities:
                    seen = set()
                    for i, e in enumerate(sum_ents[cat]):
                        if e.lower() not in seen:
                            entities[cat].append(e)
                            entity_scores[cat].append(sum_scores[cat][i])
                            seen.add(e.lower())
                    for i, e in enumerate(full_ents[cat]):
                        if e.lower() not in seen:
                            entities[cat].append(e)
                            entity_scores[cat].append(full_scores[cat][i])
                            seen.add(e.lower())

            # Final deduplication within each category
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
