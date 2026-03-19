"""
airbnb_keyword_helper.py
------------------------
Enhanced keyword extraction for Airbnb listings.

Takes the raw listing data (name, description, neighbourhood) and produces
high-quality "vibe" keywords optimised for the music recommendation pipeline.

Improvements over the base nlp_pipeline extraction:
  1. Named Entity Recognition  – extracts location names, landmarks, venues
     using a HuggingFace NER model (dslim/bert-base-NER).
  2. Comprehensive amenity blocklist – filters ~300 functional Airbnb terms.
  3. Enhanced vibe extraction – standalone atmosphere adjectives & nouns via
     POS tagging with smarter filtering.
  4. Listing metadata integration – uses listing name + neighbourhood, not
     just the description.
  5. Semantic vibe scoring – ranks candidate keywords by cosine similarity
     to vibe-anchor phrases using sentence-transformers.

Public API
----------
    enhance_keywords(listing_data, nlp_keywords, top_n=12) -> list[str]
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords

# ── Lazy-loaded heavy models ────────────────────────────────────────────────
_ner_pipeline = None
_st_model = None

_lemmatizer = WordNetLemmatizer()
_base_stopwords = set(stopwords.words("english"))


# ═══════════════════════════════════════════════════════════════════════════
# 1. COMPREHENSIVE AMENITY / FUNCTIONAL BLOCKLIST
# ═══════════════════════════════════════════════════════════════════════════

AMENITY_BLOCKLIST = {
    # ── Bedding & bedroom ──
    "bed", "beds", "bedroom", "bedrooms", "bunk", "mattress", "mattresses",
    "pillow", "pillows", "sheet", "sheets", "linen", "linens", "duvet",
    "blanket", "blankets", "comforter", "quilt", "queen", "king", "twin",
    "single", "double", "crib", "cot", "futon", "sofa bed",
    # ── Bathroom ──
    "bathroom", "bathrooms", "bath", "shower", "bathtub", "toilet",
    "towel", "towels", "shampoo", "conditioner", "soap", "body wash",
    "hair dryer", "hairdryer", "bidet",
    # ── Kitchen & dining ──
    "kitchen", "kitchenette", "oven", "stove", "microwave", "refrigerator",
    "fridge", "freezer", "dishwasher", "toaster", "blender", "coffee maker",
    "coffeemaker", "kettle", "dishes", "silverware", "dinnerware",
    "cookware", "utensils", "pots", "pans", "plates", "cups", "glasses",
    "wine glasses", "baking sheet", "cutting board",
    # ── Laundry ──
    "washer", "dryer", "laundry", "iron", "ironing board", "clothesline",
    "detergent", "washing machine",
    # ── Tech & entertainment ──
    "wifi", "internet", "ethernet", "tv", "television", "cable",
    "streaming", "roku", "chromecast", "dvd", "bluetooth", "speaker",
    "monitor", "computer", "laptop", "printer", "smart tv",
    # ── Climate ──
    "ac", "air conditioning", "heating", "heater", "fan", "fans",
    "portable fan", "ceiling fan", "radiator", "thermostat",
    "air conditioner", "central air",
    # ── Safety ──
    "smoke alarm", "smoke detector", "carbon monoxide", "fire extinguisher",
    "first aid", "fire alarm", "security camera", "deadbolt", "lock",
    "lockbox", "keypad", "safe",
    # ── Furniture ──
    "sofa", "couch", "chair", "chairs", "desk", "table", "dresser",
    "closet", "wardrobe", "shelf", "shelves", "bookshelf", "nightstand",
    "ottoman", "recliner", "dining table", "high chair",
    # ── Outdoor functional ──
    "grill", "bbq", "barbecue", "patio", "deck", "balcony",
    "outdoor furniture", "lawn", "hose",
    # ── Parking & transport ──
    "parking", "garage", "driveway", "carport", "ev charger",
    "street parking",
    # ── Building / property types ──
    "apartment", "condo", "condominium", "house", "townhouse", "cabin",
    "cottage", "villa", "studio", "loft", "duplex", "flat", "suite",
    "unit", "floor", "story", "level", "building", "complex",
    "rental", "property", "listing",
    # ── Logistics ──
    "checkin", "check-in", "checkout", "check-out", "guest", "guests",
    "host", "stay", "night", "nights", "booking", "reservation",
    "self check-in", "self checkin", "luggage", "suitcase",
    "key", "keys", "access", "entrance", "entry", "exit", "elevator",
    "stairs", "staircase", "hallway", "lobby",
    # ── Miscellaneous functional ──
    "hangers", "hanger", "outlet", "plug", "adapter", "extension cord",
    "trash", "garbage", "recycling", "vacuum", "broom", "mop",
    "cleaning", "supplies", "essentials", "basics", "amenities",
    "extra", "complimentary", "provided", "included", "available",
    "toiletries", "baby", "infant", "childproof", "child",
    "pack n play", "baby gate", "baby monitor",
    "pets", "pet", "dog", "cat", "animals",
    "wheelchair", "accessible", "ramp",
    # ── Generic / marketing words ──
    "perfect", "great", "best", "ideal", "wonderful", "amazing",
    "awesome", "excellent", "fantastic", "lovely", "nice", "good",
    "welcome", "local", "whole", "favorite", "favourite",
    "enjoy", "experience", "offer", "offers", "featuring",
    "located", "situated", "conveniently", "easily", "just",
    "minutes", "blocks", "steps", "walking distance", "close",
    "nearby", "short", "quick",
    # ── Weak / generic nouns ──
    "place", "spot", "area", "thing", "things", "way", "time",
    "plan", "array", "link", "lot", "lots", "bit", "number",
    "option", "options", "variety", "selection", "range",
    "need", "needs", "everything", "something", "nothing",
    "anyone", "everyone", "someone",
    # ── Airbnb-specific ──
    "superhost", "airbnb", "vrbo", "host", "review", "reviews",
    "rating", "star", "stars", "registration", "permit", "license",
    "legal", "registered", "verified",
}

# Amenity phrases that should be blocked as complete multi-word units
AMENITY_PHRASE_BLOCKLIST = {
    "hair dryer", "smoke alarm", "carbon monoxide", "fire extinguisher",
    "first aid", "coffee maker", "air conditioning", "high chair",
    "smart tv", "dining table", "queen bed", "king bed", "bunk bed",
    "sofa bed", "pack n play", "baby gate", "ev charger",
    "free parking", "street parking", "self check-in", "hot water",
    "long term", "board game", "board games", "baby monitor",
    "smoke detector", "fire alarm", "security camera",
    "washing machine", "ceiling fan", "portable fan",
    "full bath", "full bathroom", "hard rock",
    "international airport",
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. VIBE VOCABULARY (positive signal – boost these if found)
# ═══════════════════════════════════════════════════════════════════════════

VIBE_ADJECTIVES = {
    # ── Calm / peaceful ──
    "serene", "tranquil", "peaceful", "quiet", "calm", "relaxing",
    "soothing", "idyllic", "secluded", "private", "retreating",
    # ── Luxurious / upscale ──
    "luxurious", "elegant", "opulent", "upscale", "lavish", "posh",
    "refined", "sophisticated", "plush", "premium", "exclusive",
    "boutique", "chic", "glamorous", "swanky",
    # ── Cozy / intimate ──
    "cozy", "intimate", "warm", "charming", "quaint", "snug",
    "homey", "inviting", "welcoming", "comfortable",
    # ── Vibrant / energetic ──
    "vibrant", "lively", "energetic", "bustling", "dynamic",
    "exciting", "thrilling", "eclectic", "colorful",
    # ── Modern / stylish ──
    "modern", "contemporary", "stylish", "sleek", "minimalist",
    "trendy", "hip", "artsy", "artistic", "designer", "curated",
    # ── Historic / classic ──
    "historic", "vintage", "classic", "timeless", "antique",
    "heritage", "traditional", "colonial", "victorian",
    # ── Nature / outdoor ──
    "tropical", "beachfront", "oceanfront", "waterfront", "lakefront",
    "mountainous", "alpine", "rustic", "pastoral", "scenic",
    "panoramic", "breathtaking", "stunning", "majestic",
    # ── Urban ──
    "urban", "metropolitan", "cosmopolitan", "downtown",
    # ── Romantic ──
    "romantic", "dreamy", "enchanting", "magical", "whimsical",
}

VIBE_NOUNS = {
    # ── Nature / landscape ──
    "ocean", "beach", "sea", "sunset", "sunrise", "mountain", "mountains",
    "valley", "river", "lake", "forest", "woods", "garden", "gardens",
    "vineyard", "island", "coastline", "shore", "cliff", "waterfall",
    "meadow", "prairie", "desert", "canyon", "glacier",
    # ── Urban / city ──
    "skyline", "skyscraper", "rooftop", "terrace", "penthouse",
    "nightlife", "district", "quarter", "boulevard", "promenade",
    "plaza", "square", "avenue", "boardwalk", "harbor", "marina",
    "downtown", "midtown", "uptown",
    # ── Culture / arts ──
    "gallery", "museum", "theater", "theatre", "opera", "jazz",
    "blues", "festival", "carnival", "market", "bazaar",
    "cafe", "bistro", "restaurant", "cuisine", "gastronomy",
    "nightclub", "lounge", "bar", "pub", "brewery", "winery",
    # ── Architecture / style ──
    "mansion", "palace", "castle", "cathedral", "chapel",
    "brownstone", "townhome", "bungalow", "farmhouse", "chateau",
    "landmark", "monument", "sanctuary", "retreat", "oasis", "haven",
    "hideaway", "getaway", "paradise",
    # ── Experiences ──
    "adventure", "exploration", "safari", "trek", "hike",
    "surf", "skiing", "snowboarding", "kayak", "snorkel", "diving",
    "spa", "yoga", "meditation", "wellness",
}


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODEL LOADERS (lazy)
# ═══════════════════════════════════════════════════════════════════════════

def _get_ner_pipeline():
    """Load HuggingFace NER pipeline (dslim/bert-base-NER) on first use."""
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline as hf_pipeline
        _ner_pipeline = hf_pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def _get_st_model():
    """Get the sentence-transformer model (reuse from keyword_embedder if loaded)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


# ═══════════════════════════════════════════════════════════════════════════
# 4. EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Light cleaning: strip HTML, normalize whitespace. Preserves case & proper nouns."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_blocked(term: str) -> bool:
    """Check if a term (lowercased) matches the amenity blocklist."""
    low = term.lower().strip()
    if low in AMENITY_BLOCKLIST:
        return True
    if low in AMENITY_PHRASE_BLOCKLIST:
        return True
    # Check individual words against blocklist
    words = low.split()
    if len(words) == 1 and low in AMENITY_BLOCKLIST:
        return True
    return False


def extract_named_entities(text: str) -> dict[str, list[str]]:
    """
    Run NER on text and return entities grouped by type.

    Returns dict with keys:
        "locations"  – LOC entities (cities, regions, landmarks)
        "orgs"       – ORG entities (venues, companies, institutions)
        "misc"       – MISC entities (other notable proper nouns)
    """
    if not text:
        return {"locations": [], "orgs": [], "misc": []}

    ner = _get_ner_pipeline()
    # NER models have a max token limit; chunk long text
    chunks = _chunk_text(text, max_chars=450)

    entities: dict[str, list[str]] = {"locations": [], "orgs": [], "misc": []}
    seen = set()

    # Batch all chunks through pipeline at once for efficiency
    all_results = ner(chunks, batch_size=len(chunks))
    # If single chunk, wrap in list for uniform handling
    if chunks and isinstance(all_results[0], dict):
        all_results = [all_results]

    for results in all_results:
        for ent in results:
            word = ent["word"].strip()
            # Clean up subword tokens
            word = re.sub(r"\s*##\s*", "", word)
            word = word.strip(".,;:!?()[]{}\"'")
            label = ent["entity_group"]
            score = ent["score"]

            if len(word) < 2 or score < 0.70:
                continue
            if word.lower() in seen:
                continue
            if _is_blocked(word):
                continue

            seen.add(word.lower())
            if label == "LOC":
                entities["locations"].append(word)
            elif label == "ORG":
                entities["orgs"].append(word)
            elif label == "MISC":
                entities["misc"].append(word)
            # PER entities are skipped (host names aren't useful)

    return entities


def _chunk_text(text: str, max_chars: int = 450) -> list[str]:
    """Split text into chunks that respect sentence boundaries."""
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text[:max_chars]]


def extract_vibe_adjectives(text: str) -> list[str]:
    """Extract adjectives from text that convey atmosphere/mood."""
    if not text:
        return []

    clean = _clean_text(text).lower()
    clean = re.sub(r"[^\w\s]", " ", clean)
    tokens = word_tokenize(clean)
    tokens = [t for t in tokens if len(t) > 2 and t not in _base_stopwords]
    tagged = pos_tag(tokens)

    vibes = []
    for word, tag in tagged:
        if tag.startswith("JJ"):
            lemma = _lemmatizer.lemmatize(word, pos="a")
            if lemma in VIBE_ADJECTIVES and not _is_blocked(lemma):
                vibes.append(lemma)

    return vibes


def extract_vibe_nouns(text: str) -> list[str]:
    """Extract nouns from text that convey atmosphere/setting."""
    if not text:
        return []

    clean = _clean_text(text).lower()
    clean = re.sub(r"[^\w\s]", " ", clean)
    tokens = word_tokenize(clean)
    tokens = [t for t in tokens if len(t) > 2 and t not in _base_stopwords]
    tagged = pos_tag(tokens)

    nouns = []
    for word, tag in tagged:
        if tag.startswith("NN"):
            lemma = _lemmatizer.lemmatize(word, pos="n")
            if lemma in VIBE_NOUNS and not _is_blocked(lemma):
                nouns.append(lemma)

    return nouns


def extract_adj_noun_vibes(text: str) -> list[str]:
    """
    Extract adj-noun phrases that pass vibe filtering.
    Reuses the same logic as nlp_pipeline but with enhanced filtering.
    """
    if not text:
        return []

    clean = _clean_text(text).lower()
    clean = re.sub(r"[^\w\s]", " ", clean)
    tokens = word_tokenize(clean)
    tokens = [t for t in tokens if len(t) > 2 and t not in _base_stopwords]
    tagged = pos_tag(tokens)

    phrases = []
    for i in range(len(tagged) - 1):
        w1, t1 = tagged[i]
        w2, t2 = tagged[i + 1]
        if t1.startswith("JJ") and t2 in {"NN", "NNS"}:
            w1 = _lemmatizer.lemmatize(w1, pos="a")
            w2 = _lemmatizer.lemmatize(w2, pos="n")
            phrase = f"{w1} {w2}"
            if not _is_blocked(phrase) and not _is_blocked(w1) and not _is_blocked(w2):
                phrases.append(phrase)

    return phrases


# ═══════════════════════════════════════════════════════════════════════════
# 5. SEMANTIC VIBE SCORING
# ═══════════════════════════════════════════════════════════════════════════

_VIBE_ANCHORS = [
    "the mood and atmosphere of a place",
    "the feeling and vibe of a travel destination",
    "emotional tone of a neighborhood",
    "cultural spirit of a city",
    "the energy and character of a location",
]

_ANTI_ANCHORS = [
    "household appliances and amenities",
    "furniture and bedding items",
    "property management and booking logistics",
    "cleaning supplies and toiletries",
]

_vibe_anchor_embs = None
_anti_anchor_embs = None


def _get_anchor_embeddings():
    """Compute and cache anchor embeddings for vibe scoring."""
    global _vibe_anchor_embs, _anti_anchor_embs
    if _vibe_anchor_embs is None:
        model = _get_st_model()
        _vibe_anchor_embs = model.encode(_VIBE_ANCHORS, normalize_embeddings=True)
        _anti_anchor_embs = model.encode(_ANTI_ANCHORS, normalize_embeddings=True)
    return _vibe_anchor_embs, _anti_anchor_embs


def score_keywords_by_vibe(keywords: list[str]) -> list[tuple[str, float]]:
    """
    Score each keyword by semantic similarity to vibe anchors vs anti-anchors.
    Returns list of (keyword, vibe_score) sorted by score descending.
    vibe_score > 0 means more vibe-like, < 0 means more amenity-like.
    """
    if not keywords:
        return []

    model = _get_st_model()
    vibe_embs, anti_embs = _get_anchor_embeddings()

    kw_embs = model.encode(keywords, normalize_embeddings=True)

    # Mean similarity to vibe anchors
    vibe_sims = (kw_embs @ vibe_embs.T).mean(axis=1)
    # Mean similarity to anti-anchors
    anti_sims = (kw_embs @ anti_embs.T).mean(axis=1)
    # Net vibe score
    net_scores = vibe_sims - anti_sims

    scored = list(zip(keywords, net_scores.tolist()))
    scored.sort(key=lambda x: -x[1])
    return scored


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def enhance_keywords(
    listing_data: Dict[str, Optional[str]],
    nlp_keywords: list[str] | None = None,
    top_n: int = 12,
) -> dict:
    """
    Produce enhanced vibe keywords for a listing.

    Parameters
    ----------
    listing_data : dict with keys "name", "description", "neighbourhood_cleansed"
    nlp_keywords : keywords already extracted by nlp_pipeline (will be included
                   if they pass vibe scoring)
    top_n        : max number of keywords to return

    Returns
    -------
    dict with keys:
        "keywords"       : list[str] — all deduplicated, vibe-ranked keywords
        "location_terms" : list[str] — NER-extracted location/landmark names
                           (for direct use in lyrics search layer)
    """
    name = _clean_text(str(listing_data.get("name", "") or ""))
    description = _clean_text(str(listing_data.get("description", "") or ""))
    neighbourhood = str(listing_data.get("neighbourhood_cleansed", "") or "").strip()

    # Combine all text for extraction
    full_text = f"{name}. {description}"

    # ── Stage 1: Named Entity Recognition ────────────────────────────────
    entities = extract_named_entities(full_text)
    location_terms = entities["locations"] + entities["orgs"] + entities["misc"]

    # Add neighbourhood as a location term
    if neighbourhood and not _is_blocked(neighbourhood):
        location_terms.append(neighbourhood)

    # Deduplicate location terms
    seen_loc = set()
    deduped_locations: list[str] = []
    for t in location_terms:
        key = t.lower().strip()
        if key and key not in seen_loc:
            seen_loc.add(key)
            deduped_locations.append(t.strip())
    location_terms = deduped_locations

    # ── Stage 2: Vibe adjectives & nouns ─────────────────────────────────
    vibe_adjs = extract_vibe_adjectives(full_text)
    vibe_nouns = extract_vibe_nouns(full_text)
    adj_noun_phrases = extract_adj_noun_vibes(full_text)

    # ── Stage 3: Include original nlp_pipeline keywords ──────────────────
    original = list(nlp_keywords) if nlp_keywords else []

    # ── Stage 4: Collect all candidates ──────────────────────────────────
    # Location terms get priority (they drive lyrics search)
    # Deduplicate while preserving order
    seen = set()
    all_candidates: list[str] = []

    def _add(term: str):
        key = term.lower().strip()
        if key and key not in seen and len(key) > 1:
            seen.add(key)
            all_candidates.append(term.strip())

    # Priority 1: Location entities (proper nouns, landmarks)
    for t in location_terms:
        _add(t)

    # Priority 2: Adj-noun vibe phrases
    for t in adj_noun_phrases:
        _add(t)

    # Priority 3: Vibe adjectives (most common first)
    adj_counts = Counter(vibe_adjs)
    for adj, _ in adj_counts.most_common():
        _add(adj)

    # Priority 4: Vibe nouns (most common first)
    noun_counts = Counter(vibe_nouns)
    for noun, _ in noun_counts.most_common():
        _add(noun)

    # Priority 5: Original nlp_pipeline keywords
    for t in original:
        _add(t)

    if not all_candidates:
        return {"keywords": original[:top_n], "location_terms": location_terms}

    # ── Stage 5: Semantic vibe scoring & final ranking ───────────────────
    scored = score_keywords_by_vibe(all_candidates)

    # Keep all location entities regardless of vibe score (they drive lyrics layer)
    location_set = {t.lower() for t in location_terms}
    final: list[str] = []
    used = set()

    # First: add location terms (they're critical for finding songs about a place)
    for kw, score in scored:
        if kw.lower() in location_set and kw.lower() not in used:
            final.append(kw)
            used.add(kw.lower())

    # Then: add vibe terms ranked by score, filtering low-scoring ones
    for kw, score in scored:
        if kw.lower() not in used and score > -0.05:
            final.append(kw)
            used.add(kw.lower())
        if len(final) >= top_n:
            break

    return {
        "keywords": final[:top_n],
        "location_terms": location_terms,
    }
