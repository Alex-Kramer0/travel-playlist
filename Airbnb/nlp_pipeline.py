'''
HOW TO CALL FUNCTION: 
from Airbnb.nlp_pipeline import analyze_listing_from_url

## Unzip the output file prior to running! 
DATASET_PATH = "Airbnb/data/Output.csv"
NRC_PATH = "Airbnb/emolex/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# sample listing
listing_url = "https://www.airbnb.com/rooms/2536175"

keywords_json, emotions_json = analyze_listing_from_url(
    url=listing_url,
    dataset_path=DATASET_PATH,
    nrc_path=NRC_PATH,
)

print("KEYWORDS JSON")
print(keywords_json)

print("\nEMOTIONS JSON")
print(emotions_json)
'''

# =========================
# IMPORTS
# =========================
import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords


# =========================
# NLTK SETUP
# =========================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

lemmatizer = WordNetLemmatizer()
base_stopwords = set(stopwords.words("english"))


# =========================
# CONFIG
# =========================
CITY_BIGRAMS = {
    "fort lauderdale",
}

PLACE_TOKENS = {
    "bozeman",
    "lauderdale",
    "hallandale",
    "hollywood",
    "olas",
    "isles",
    "bay",
    "waterway",
    "florida",
    "dania",
    "intracoastal",
}

AIRBNB_FUNCTIONAL = {
    "bedroom",
    "bathroom",
    "wifi",
    "kitchen",
    "parking",
    "apartment",
    "condo",
    "house",
    "guest",
    "guests",
    "stay",
    "checkin",
    "checkout",
    "night",
    "nights",
    "queen",
    "king",
    "tv",
    "internet",
    "ac",
}

UTILITY_NOUNS = {
    "pool",
    "tub",
    "washer",
    "dryer",
    "bed",
    "beds",
    "room",
    "entrance",
    "cabana",
    "cabanas",
    "access",
    "speed",
    "chair",
    "ceiling",
    "closet",
    "sofa",
    "bath",
    "bathroom",
    "airport",
}

MARKETING_WORDS = {
    "perfect",
    "local",
    "whole",
    "best",
    "great",
    "ideal",
    "welcome",
}

WEAK_NOUNS = {
    "plan",
    "enjoy",
    "spot",
    "place",
    "thing",
    "area",
    "array",
    "link",
}

EXACT_PHRASE_BLOCKLIST = {
    "hard_rock",
    "full_bath",
    "full_bathroom",
    "smart_tv",
    "international_airport",
    "fll_airport",
}

ALL_STOPWORDS = (
    base_stopwords
    .union(AIRBNB_FUNCTIONAL)
    .union(MARKETING_WORDS)
    .union(PLACE_TOKENS)
)

EMOTIONS_8 = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]

POLARITY = ["positive", "negative"]
ALL_NRC_CATS = EMOTIONS_8 + POLARITY


# =========================
# DATA LOADING
# =========================
def load_listing_dataset(dataset_path: str) -> pd.DataFrame:
    '''
    function: loads the airbnb dataset
    param:
        dataset_path: the path to the airbnb dataset formatted as an csv
    return: a dataframe of the airbnb dataset
    '''
    return pd.read_csv(dataset_path)


def get_listing_by_url(url: str, df: pd.DataFrame) -> Dict[str, Optional[str]]: #url optional 
    '''
    function: finds inputted URL within the dataset
    param: 
        url: the listing url that was inputted by the user
        df: the dataframe containing all of the airbnb listings
    return: listing ID, listing URL, name, description, and neighbourhood for 
            the found listing
    '''
    match = df[df["listing_url"] == url]

    # if no match, raise error
    if match.empty:
        raise ValueError(f"URL not found in dataset: {url}")

    row = match.iloc[0]

    return {
        "id": row.get("id"),
        "listing_url": row.get("listing_url"),
        "name": row.get("name"),
        "description": row.get("description"),
        "neighbourhood_cleansed": row.get("neighbourhood_cleansed"),
    }


# =========================
# TEXT CLEANING
# =========================
def normalize_text(text: str) -> str:
    '''
    function: normalizes the text for the listing description 
    param:
        text: a string containing the listing descriptions
    return: 
        text: a cleaned version of the listing description 
    '''

    # handles missing values
    if pd.isna(text) or text is None: 
        return ""

    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text) # remove HTML tags 
    text = text.encode("ascii", "ignore").decode()  # removes  non-ASCII characters

    # remove location 
    for city in CITY_BIGRAMS:
        text = text.replace(city, " ")


    text = text.translate(str.maketrans("", "", string.punctuation))# remove punctuations
    text = re.sub(r"\d+", " ", text) # remove numbers
    text = re.sub(r"\s+", " ", text).strip() # remove excess spaces

    return text


def tokenize_for_emotion(text: str) -> List[str]:
    """
    function: prepares text for emotion analysis by converting it into tokens 
    params:
        text: clean listing descriptions
    returns: 
        tokens: a list of cleaned tokens for emotion analysis
    """
    text = normalize_text(text) # call text cleaning function 
    if not text:
        return []

    # tokenize 
    # ex: "beautiful apartment downtown" -> ["beautiful", "apartment", "downtown"]
    tokens = word_tokenize(text)

    filtered_tokens = []
    for token in tokens:
        if len(token) > 2:
            filtered_tokens.append(token)

    lemmatized_tokens = []
    for token in filtered_tokens:
        lemma = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemma)

    return lemmatized_tokens


# =========================
# KEYWORD EXTRACTION
# =========================
def extract_adj_noun_phrases(text: str) -> List[str]:
    '''
    function: extracts adjective nouns phrase pairs from text 
            Ex. extracts 'ocean view'
    params:
        text: takes in airbnb description 
    return: 
        phrases: a list of the adjective and nouns 
    '''

    text = normalize_text(text)
    if not text:
        return []

    # clean text into word tokens
    tokens = word_tokenize(text)

    filtered_tokens = []

    for token in tokens:
        if len(token) > 2 and token not in ALL_STOPWORDS:
            filtered_tokens.append(token)

    tokens = filtered_tokens
    # part of speech tagging - label each word by grammar role
    tagged = pos_tag(tokens)

    phrases = []
    for i in range(len(tagged) - 1):
        word_1, tag_1 = tagged[i]
        word_2, tag_2 = tagged[i + 1]

        #JJ = adective, NN = singular noun, NNS = plurual noun 
        if tag_1.startswith("JJ") and tag_2 in {"NN", "NNS"}:
            word_1 = lemmatizer.lemmatize(word_1)
            word_2 = lemmatizer.lemmatize(word_2)
            phrases.append(f"{word_1}_{word_2}")

    return phrases


def keep_vibe_phrase(phrase: str) -> bool:
    '''
    function: determines if the extracted adjective-noun phrase needs to be excluded or included
    params:
        phrase: phrases formatted as adjective_noun 
    returns: 
        bool: true if the phrase should be kept, false if it should be filtered out
    '''
    # remove phrases that are blocekd 
    if phrase in EXACT_PHRASE_BLOCKLIST:
        return False

    # split the phrases
    parts = phrase.split("_")
    if len(parts) != 2:
        return False

    adj, noun = parts

    # remove location-related words
    if adj in PLACE_TOKENS or noun in PLACE_TOKENS:
        return False

    # remove 'wifi' 'free parking' etc
    if noun in UTILITY_NOUNS:
        return False

    if noun in WEAK_NOUNS:
        return False

    if adj in MARKETING_WORDS:
        return False

    return True


def extract_vibe_keywords(text: str, top_n: int = 5) -> List[str]:
    '''
    function: extracts the most common vibe phrases from the text  
    params: 
        text: raw listing descriptions

    returns: a list of the most frequent vibe phrases as reasable text
    '''
    phrases = extract_adj_noun_phrases(text)
    filtered_phrases = []

    for phrase in phrases:
        if keep_vibe_phrase(phrase):
            filtered_phrases.append(phrase)

    phrases = filtered_phrases
    top_phrases = []

    phrase_counts = Counter(phrases)
    most_common_phrases = phrase_counts.most_common(top_n)

    for phrase, count in most_common_phrases:
        top_phrases.append(phrase)
    return [phrase.replace("_", " ") for phrase in top_phrases]


# =========================
# NRC LOADING
# =========================
def load_nrc_lexicon(nrc_path: str) -> Dict[str, set]:
    """
    function: loads the emolex lexicon used for emotion analysis and converts it into a dictornary 
    mapping emotions to associated words
    params: 
        nrc_path: path to the NRC dataset
    returns: a dictorary where the key = emotion category and 
            value = set of words associaed with that emotion
    """

    with open(nrc_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = []

        for line in f:
            stripped_line = line.strip()

            if stripped_line and not stripped_line.startswith("#"):
                lines.append(stripped_line)

    if not lines:
        raise ValueError("NRC file appears empty after removing comments and blank lines.")

    sample = lines[0]
    if "\t" in sample:
        sep = "\t"
    elif "," in sample:
        sep = ","
    elif ";" in sample:
        sep = ";"
    else:
        sep = None

    cat_to_words = defaultdict(set)

    def add_pair(word: str, category: str, association: str = "1") -> None:
        '''
        helper function to add valid word-emotion pairs 
        '''
        word = word.strip().lower()
        category = category.strip().lower()
        association = association.strip()

        if category in ALL_NRC_CATS and association in {"1", "1.0", "true", "True"}:
            cat_to_words[category].add(word)

    for line in lines:
        parts = line.split(sep) if sep else line.split()

        if len(parts) >= 3:
            a, b, c = parts[0], parts[1], parts[2]

            if b.lower() in ALL_NRC_CATS:
                add_pair(a, b, c)
            elif a.lower() in ALL_NRC_CATS:
                add_pair(b, a, c)

        elif len(parts) == 2:
            a, b = parts[0], parts[1]

            if b.lower() in ALL_NRC_CATS:
                add_pair(a, b, "1")
            elif a.lower() in ALL_NRC_CATS:
                add_pair(b, a, "1")

    if not cat_to_words:
        raise ValueError("Could not parse NRC lexicon file.")

    return dict(cat_to_words)


# =========================
# EMOTION SCORING
# =========================
def score_nrc(tokens: List[str], cat_to_words: Dict[str, set]) -> Dict[str, Optional[float]]:
    '''
    functions: scores a list of tokens using the NRC Emolex
    params: 
        tokens: tokenized + cleaned words from the text
        cat_to_words: dictionary mapping emotion categories
    returns: dictionary containg: count for each emotion, normalized emotion scores, number
            of tokens, dominant emotion, positive to negative ratio 
    '''
    scores = {category: 0 for category in ALL_NRC_CATS}

    if not tokens:
        scores["n_tokens"] = 0
        scores["dominant_emotion"] = None
        scores["pos_neg_ratio"] = None

        for category in ALL_NRC_CATS:
            scores[f"{category}_norm"] = 0.0

        return scores

    for token in tokens:
        for category in ALL_NRC_CATS:
            if token in cat_to_words.get(category, set()):
                scores[category] += 1

    n_tokens = len(tokens)
    scores["n_tokens"] = n_tokens

    for category in ALL_NRC_CATS:
        scores[f"{category}_norm"] = scores[category] / n_tokens

    emotion_counts = {}

    for emotion in EMOTIONS_8:
        emotion_counts[emotion] = scores[emotion]

    max_count = max(emotion_counts.values()) if emotion_counts else 0
    scores["dominant_emotion"] = None if max_count == 0 else max(emotion_counts, key=emotion_counts.get)

    scores["pos_neg_ratio"] = (scores["positive"] + 1) / (scores["negative"] + 1)

    return scores


def extract_emotions(text: str, cat_to_words: Dict[str, set]) -> Dict[str, Optional[float]]:
    ''' 
        function: extracts emotion scores from text
        param: 
            text: description
            cat_to_words: dictionary mapping emotion categories to a set of word
                        associations
        return:
            dictionary: containing emotion counts, normalized scores, dominant emotion, token count
                        and postive/negative ration
    '''
    tokens = tokenize_for_emotion(text)
    return score_nrc(tokens, cat_to_words)


# =========================
# JSON BUILDERS
# =========================
def build_keyword_json(listing_data: Dict[str, Optional[str]], keywords: List[str]) -> Dict:
    '''
    function: creates a structured dictionary containing the needed metadata and the keywords
    return: returns a JSON-stle ductionary 
    '''
    return {
        "id": listing_data.get("id"),
        "listing_url": listing_data.get("listing_url"),
        "name": listing_data.get("name"),
        "neighbourhood_cleansed": listing_data.get("neighbourhood_cleansed"),
        "keywords": keywords,
    }


def build_emotion_json(listing_data: Dict[str, Optional[str]], emotion_scores: Dict[str, Optional[float]]) -> Dict:
    '''
    function: creates a structured dictionary containing the needed metadata and the emotion scoring
    return: returns a JSON-stle dictionary 
    '''
    return {
        "id": listing_data.get("id"),
        "listing_url": listing_data.get("listing_url"),
        "name": listing_data.get("name"),
        "neighbourhood_cleansed": listing_data.get("neighbourhood_cleansed"),
        "dominant_emotion": emotion_scores.get("dominant_emotion"),
        "emotion_scores": {
            emotion: emotion_scores.get(f"{emotion}_norm", 0.0)
            for emotion in EMOTIONS_8
        },
        "positive_norm": emotion_scores.get("positive_norm", 0.0),
        "negative_norm": emotion_scores.get("negative_norm", 0.0),
        "pos_neg_ratio": emotion_scores.get("pos_neg_ratio"),
    }


# =========================
# MAIN PIPELINE
# =========================
def analyze_listing_from_url(url: str, dataset_path: str, nrc_path: str) -> Tuple[Dict, Dict]:
    '''
    function: runs the full pipeline 
    '''
    listings_df = load_listing_dataset(dataset_path)
    cat_to_words = load_nrc_lexicon(nrc_path)

    listing_data = get_listing_by_url(url, listings_df)
    description = listing_data.get("description", "") or ""

    keywords = extract_vibe_keywords(description, top_n=5)
    emotions = extract_emotions(description, cat_to_words)

    keyword_json = build_keyword_json(listing_data, keywords)
    emotion_json = build_emotion_json(listing_data, emotions)

    return keyword_json, emotion_json


## alex create new branch accepting descriptions for the input