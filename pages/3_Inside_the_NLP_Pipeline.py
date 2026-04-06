
import streamlit as st
import pandas as pd

st.set_page_config(page_title="NLP Pipeline", layout="wide")

# ------------Helper functions-----------------
def section_header(title: str, subtitle: str | None = None):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(subtitle)


def method_card(title, why_tried, worked, limitations, takeaway):
    st.markdown(f"### {title}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Why we tried it**")
        st.write(why_tried)

        st.markdown("**What worked**")
        st.write(worked)

    with col2:
        st.markdown("**Limitations**")
        st.write(limitations)

        st.markdown("**Takeaway**")
        st.write(takeaway)


# -----------------------------Intro-----------------------------
st.title("Inside the NLP Pipeline")
st.markdown(
    """
This page explains how the NLP pipeline evolved from baseline keyword extraction methods
to a more semantically meaningful workflow that powers playlist generation from Airbnb
listing descriptions.
"""
)

st.info(
    """
**Pipeline overview:** Airbnb Description → POS Keyword Extraction → Zero-Shot Emotion Classification
→ Handoff to Spotify Pipeline
"""
)

st.divider()


# -----------------------------Problem Statement
section_header(
    "The Problem",
    """
We were not just trying to extract *frequent* or *unique* words — we were trying to
capture the **vibe** of a listing.

That meant identifying descriptive, experiential, and atmosphere-related language,
even when those words or phrases were common across listings in the same city.
"""
)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### Design Goals")
    st.markdown(
        """
- Capture listing **vibe**, not just statistical uniqueness  
- Preserve meaningful phrases and descriptive language  
- Reduce generic filler terms  
- Produce outputs that can be used downstream for **emotion classification** and **playlist generation**
"""
    )

with col2:
    st.markdown("### Why TF-IDF Alone Wasn't Enough")
    st.warning(
        """
TF-IDF is useful for finding distinctive words quickly, but our use case required
surfacing words and phrases that may still be important **even if they are common**.

For example, a phrase like **“ocean view”** may appear often in a Florida market,
but it is still central to the listing's atmosphere.
"""
    )

st.divider()


# -----------------------------Methods tested-----------------------------
section_header(
    "Methods We Tested",
    "Below is the progression of techniques we tried while refining the NLP pipeline."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "TF-IDF Keyword Extraction",
        "POS + TF-IDF Mixed Approach",
        "POS Keyword Extraction",
        "NRC EmoLex",
        "Zero-Shot Classification",
        "Comparison Table",
    ]
)

with tab1:
    method_card(
        title="TF-IDF",
        why_tried="""
We started with TF-IDF as a baseline method for identifying prominent words and phrases
in listing descriptions.
""",
        worked="""
- Quick to implement
- Easy to compare unigram vs bigram extraction
- Useful as an initial baseline
""",
        limitations="""
- Unigrams were too generic (for example: town, location, great)
- Bigrams were more specific but sparse and noisy
- Combined n-grams were the most interpretable, but still not aligned with the main goal
- Biased toward statistical rarity rather than experiential relevance
""",
        takeaway="""
TF-IDF helped us understand the text, but it was not sufficient for vibe extraction.
It was better as an exploratory baseline than a final solution.
""",
    )

    st.markdown("#### TF-IDF Notes")
    st.markdown(
        """
- **Unigram:** generic (town, location, great)  
- **Bigram:** sparse and busy  
- **Combined:** best results but still not what we were looking for  
- We compared unigram and bigram TF-IDF extraction to evaluate whether multi-word phrases better capture listing vibe  
- Bigrams improved contextual specificity (for example, *river rock fireplace*) but increased sparsity  
- A combined n-gram approach gave the best interpretability among TF-IDF variants  
"""
    )

with tab2:
    method_card(
        title="POS Filtering + TF-IDF",
        why_tried="""
To improve the quality of extracted keywords, we filtered the text by part of speech
before applying TF-IDF. This helped emphasize more descriptive terms such as nouns
and adjectives.
""",
        worked="""
- Significantly closer to what we wanted
- Reduced some generic filler language
- Improved interpretability of results
""",
        limitations="""
- Still inherited TF-IDF's tendency to prioritize rarity
- Still missed common but important vibe phrases
- Better than the starting point, but not fully aligned with the project goal
""",
        takeaway="""
This was a meaningful improvement and showed that linguistic filtering helped,
but the weighting logic still limited performance.
""",
    )

with tab3:
    method_card(
        title="POS Filtering Keyword Extraction",
        why_tried="""
We tested a simpler keyword extraction approach focused on retaining structurally
meaningful descriptive words and phrases without relying as heavily on corpus rarity.
""",
        worked="""
- Simpler and easier to interpret
- Better keyword quality
- More aligned with human intuition
- Produced cleaner results
""",
        limitations="""
- May still require hand-tuning depending on text variation
- Rule-based logic can miss subtle context or implied meaning
""",
        takeaway="""
This approach became one of the two final components in the pipeline because it gave
clean, interpretable vibe descriptors with less complexity than TF-IDF-based approaches.
""",
    )

with tab4:
    method_card(
        title="NRC EmoLex",
        why_tried="""
We explored NRC EmoLex as a lexicon-based way to estimate emotional tone by matching
words in a listing description to a predefined dictionary of emotion-associated terms.
""",
        worked="""
- Easy to understand and explain
- Lightweight compared with model-based methods
- Useful for testing whether listing descriptions contain emotion-linked language
- Provides a transparent, dictionary-based baseline for emotion extraction
""",
        limitations="""
- Only detects emotions when the exact or closely related lexicon words appear
- Struggles with context, nuance, and multi-word meaning
- Misses implied emotional tone when descriptions are descriptive rather than explicitly emotional
- Less flexible than model-based classification for this use case
""",
        takeaway="""
NRC EmoLex was useful as a baseline for emotion extraction, but it was not strong enough
to serve as the final emotion layer for Airbnb vibe interpretation.
""",
    )

    st.markdown("#### How NRC EmoLex Works")
    st.markdown(
        """
NRC EmoLex is a **lexicon-based emotion detection method**. It uses a predefined dictionary
where individual words are associated with one or more emotions such as **joy, trust,
anticipation, fear, sadness, anger, disgust, and surprise**.

In practice, the method:
- tokenizes the text into words
- checks whether each word appears in the NRC emotion lexicon
- retrieves the emotion tags associated with matched words
- aggregates those matches to produce an overall emotional profile for the text

This makes the method transparent and interpretable, but it also means the quality of the
output depends heavily on exact word overlap with the lexicon. For Airbnb descriptions,
that often limited its ability to capture implied mood or atmosphere.
"""
    )

with tab5:
    method_card(
        title="Zero-Shot Emotion Classification",
        why_tried="""
We needed a way to extract the emotional tone of a listing without hand-labeling a training
dataset. Zero-shot classification let us test whether a description aligned with candidate
emotion labels such as calm, cozy, joyful, adventurous, or peaceful.
""",
        worked="""
- Captured emotional tone more flexibly than a dictionary lookup
- Worked even when the text implied an emotion without explicitly naming it
- No custom labeled dataset required
- Scaled well across different listing styles and cities
""",
        limitations="""
- Depends on the design and quality of candidate labels
- Can be more computationally expensive than lexicon-based methods
- Outputs still require interpretation and validation
""",
        takeaway="""
Zero-shot classification became the final emotion method because it better captured the
felt atmosphere of the descriptions, not just isolated emotion words.
""",
    )

    st.markdown("#### What Zero-Shot Classification Is Actually Doing")
    st.markdown(
        """
In this project, zero-shot classification is **not primarily being used to generate vibe keywords**.
It is being used to **classify the emotional tone of the listing text**.

Instead of looking for exact emotion words, the model compares the full listing description
against a set of candidate emotion labels and estimates which emotions best fit the text
based on contextual meaning.

That means it can identify emotional themes such as:
- calm
- cozy
- adventurous
- romantic
- peaceful

even when those exact words do not appear in the description. This made it a stronger fit
than NRC EmoLex for our final pipeline because it captures **implied emotion and atmosphere**
rather than just direct word matches.
"""
    )

with tab6:
    comparison_df = pd.DataFrame(
        {
            "Method": [
                "TF-IDF",
                "POS + TF-IDF",
                "POS Keyword Extraction",
                "NRC EmoLex",
                "Zero-Shot Classification",
            ],
            "Interpretability": ["Medium", "Medium-High", "High", "High", "High"],
            "Vibe Relevance": ["Low", "Medium", "High", "Medium", "Very High"],
            "Complexity": ["Low", "Medium", "Low", "Low", "Medium"],
            "Role in Project": [
                "Baseline",
                "Refinement Step",
                "Final Core Method",
                "Tested but Not Final",
                "Final Core Method",
            ],
        }
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.divider()


# -----------------------------Example
section_header(
    "Example Across Methods",
    """
To illustrate the differences between methods, we can apply each technique to the same sample listing description and compare the outputs.
"""
)

st.markdown("### Example Listing Description")
st.markdown(
    "Rustic canyon getaway in clean-air, rural Malibu mountains! <br /><br />Private gravel entrance w parking.<br />Adjacent to incredible canyon & ocean views, singing birds, hiking.<br /><br />Quiet neighborhood for heavenly sleeps. One queen bed, One trundle bed with two single mattresses, one air mattress. A/C for summer, space heater for winter. Kitchenette (no kitchen sink) and full bath.<br /><br />Highlights!<br />Claw-foot Tub<br />Mountain Sunsets<br />Amazon Echo<br />Wild Bird families & bunnies<br />Hiking at end of road <br />2.5 miles to the beach",
    unsafe_allow_html=True
)

st.markdown("### Extraction Results by Method")

example_tab1, example_tab2, example_tab3, example_tab4 = st.tabs(
    [
        "TF-IDF Output",
        "POS + TF-IDF Output",
        "POS Keyword Output",
        "Final Pipeline Output",
    ]
)

with example_tab1:
    st.markdown("**Extracted keywords / phrases**")
    st.code(
        """
TF-IDF Unigrams Keywords:
- one
- mattress
- hiking
- mountain
- bird

TF-IDF Bigrams Keywords:
- winer kitchenette
- entrance adjacent
- kitchenette sink
- incredible canyon
- hiking quiet

TF-IDF Unigrams + Bigrams Keywords:
- one
- canyon
- mattress
- mountain
- bird
""",
        language="text",
    )
    st.markdown("**Commentary**")
    st.write(
        "This output was too generic and noisy, capturing many common words that do not convey the unique vibe of the listing. We saw the strongest results with the bigram approach, but it still included irrelevant phrases and missed key atmosphere elements."
    )

with example_tab2:
    st.markdown("**Extracted keywords / phrases**")
    st.code(
        """
POS + TF-IDF Unigrams Keywords:
- wild_bird
- tub_mountain
- single_mattress
- rustic_canyon
- rural_malibu

POS + TF-IDF Bigrams Keywords:
- tub_mountain wild_bird
- single_mattresses tub_mountain
- rustic_canyon rural_malibu
- rural_malibu gravel_entrance
- quiet_neighborhood single_mattresses

POS + TF-IDF Unigrams + Bigrams Keywords:
- wild_bird
- quiet_neighborhood single_mattresses
- gravel_entrance incredible_canyon
- incredible_canyon ocean_views
- incredible_canyon
""",
        language="text",
    )
    st.markdown("**Commentary**")
    st.write(
        "TF-IDF still underperformed even after POS filtering because many Airbnb listings share similar vocabulary. While TF-IDF is useful for identifying statistically distinctive language, our goal was to preserve meaningful vibe phrases that may be common within a market, such as 'ocean view.'"
    )

with example_tab3:
    st.markdown("**Extracted keywords / phrases**")
    st.code(
        """
POS Keywords:
- mountain
- canyon
- neighborhood
- sleeps
- tub
- families
""",
        language="text",
    )
    st.markdown("**Commentary**")
    st.write(
        "This method uses a rule-based NLP pipeline that combines normalization, domain-specific stopword filtering, POS-tag-based phrase extraction, lemmatization, and heuristic pruning to identify high-signal descriptive terms representing listing ambience or vibe. It is interpretable and lightweight, though it can still miss more nuanced signals."
    )

with example_tab4:
    st.markdown("**Final POS keyword output**")
    st.code(
            """
POS Keywords:
- Rustic Canyon
- Rural Malibu
- Incredible Canyon
- Ocean Views
- Quiet Neighborhood
""",
        language="text",
        )

    st.markdown("**Playlist interpretation**")
    st.write(
        "Together, the POS keyword layer tells us what is materially present in the listing, while zero-shot emotion classification tells us how the stay is likely to feel. That combination creates a more useful handoff into playlist generation than either method alone."
    )

st.divider()


# -----------------------------Final Pipeline
section_header(
    "Final NLP Pipeline",
    "The current workflow combines interpretable keyword extraction with model-based emotion classification."
)

st.markdown(
    """
1. **Ingest Airbnb listing description**  
2. **Clean and preprocess text**  
3. **Apply POS Keyword Extraction** to identify descriptive vibe-related terms  
4. **Apply Zero-Shot Emotion Classification** to estimate the emotional tone of the listing  
5. **Pass keywords + emotions into the Spotify pipeline**    
"""
)

st.success(
    """
**Why this works:** POS keyword extraction captures concrete descriptive elements from the listing,
while zero-shot emotion classification captures the emotional atmosphere those descriptions imply.
Together, they create a richer and more useful representation for playlist generation.
"""
)

st.divider()


# -----------------------------
# Why this matters
# -----------------------------
section_header(
    "Why the NLP Layer Matters",
    """
The NLP layer is the bridge between raw listing text and a playlist that feels intentional.
Without it, the Spotify handoff would be based on unstructured copy rather than a meaningful
representation of place, mood, and guest experience.
"""
)

st.markdown("### What This Layer Contributes")
st.markdown(
    """
This layer does more than extract words from text. It transforms a free-form Airbnb description
into structured signals that downstream systems can actually use.

- **POS keyword extraction** identifies the concrete descriptive elements of the stay, such as landscape, amenities, and atmosphere cues  
- **Zero-shot emotion classification** interprets the emotional tone implied by the description, such as peaceful, cozy, romantic, or adventurous  
- Together, these signals create a more complete representation of the listing than either raw text or basic keyword frequency alone  
"""
)

st.markdown("### End-to-End Example")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Input Text**")
    st.write("Airbnb description")

with col2:
    st.markdown("**POS Keywords**")
    st.write("Descriptive place and vibe terms")

with col3:
    st.markdown("**Zero-Shot Emotions**")
    st.write("Inferred emotional tone of the stay")

with col4:
    st.markdown("**Playlist Direction**")
    st.write("Spotify recommendation logic")

st.markdown("### Why This Improves the Product")
st.markdown(
    """
- It reduces reliance on raw word frequency and instead captures **experience-level meaning**  
- It makes playlist generation more consistent across listings with very different writing styles  
- It separates **what the listing contains** from **how the listing feels**, which is useful for recommendation logic  
- It creates outputs that are more interpretable for both technical and non-technical audiences  
- It gives the downstream music pipeline cleaner, more structured inputs, which should improve playlist fit and explainability  
"""
)

st.divider()

st.markdown("### Key Takeaways")
st.markdown(
    """
- TF-IDF was useful as a baseline, but not ideal for extracting listing vibe  
- POS filtering improved quality by reducing generic language  
- POS keyword extraction became a final core method because it produced cleaner descriptive signals  
- NRC EmoLex was helpful as a lexicon-based baseline for emotion extraction, but it was too limited for the final pipeline  
- Zero-shot classification became the final emotion method because it better captured implied emotional atmosphere  
- The final pipeline uses **POS keyword extraction + Zero-Shot emotion classification**
"""
)