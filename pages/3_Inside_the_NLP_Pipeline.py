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
**Pipeline overview:** Airbnb Description → Keyword / Vibe Extraction → Emotion Scoring
(NRC EmoLex) → Handoff to Spotify Pipeline
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
- Produce outputs that can be used downstream for **emotion analysis** and **playlist generation**
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "TF-IDF",
        "POS + TF-IDF",
        "POS Keyword Extraction",
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
- Less flexible than a semantic classification approach
""",
        takeaway="""
This approach gave better results with less complexity, which made it a strong step
forward in the pipeline.
""",
    )

with tab4:
    method_card(
        title="Zero-Shot Text Classification",
        why_tried="""
We needed a way to identify higher-level vibe concepts without building a manually
labeled training dataset. Zero-shot classification allowed us to map listing text
to meaningful semantic labels.
""",
        worked="""
- Captured semantic meaning better than TF-IDF approaches
- Flexible and scalable
- No custom labeled dataset required
- Much closer to the actual vibe-extraction goal
""",
        limitations="""
- Depends on candidate label design
- Can be more computationally expensive than simpler extraction methods
- Outputs still need validation and interpretation
""",
        takeaway="""
Zero-shot classification became the strongest fit for the final NLP pipeline because
it better captured the atmosphere and intent of the descriptions.
""",
    )

with tab5:
    comparison_df = pd.DataFrame(
        {
            "Method": [
                "TF-IDF",
                "POS + TF-IDF",
                "POS Keyword Extraction",
                "Zero-Shot Classification",
            ],
            "Interpretability": ["Medium", "Medium-High", "High", "High"],
            "Vibe Relevance": ["Low", "Medium", "High", "Very High"],
            "Complexity": ["Low", "Medium", "Low", "Medium"],
            "Role in Project": [
                "Baseline",
                "Refinement Step",
                "Strong Candidate",
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
st.markdown("Rustic canyon getaway in clean-air, rural Malibu mountains! <br /><br />Private gravel entrance w parking.<br />Adjacent to incredible canyon & ocean views, singing birds, hiking.<br /><br />Quiet neighborhood for heavenly sleeps. One queen bed, One trundle bed with two single mattresses, one air mattress. A/C for summer, space heater for winter. Kitchenette (no kitchen sink) and full bath.<br /><br />Highlights!<br />Claw-foot Tub<br />Mountain Sunsets<br />Amazon Echo<br />Wild Bird families & bunnies<br />Hiking at end of road <br />2.5 miles to the beach")

st.markdown("### Extraction Results by Method")

example_tab1, example_tab2, example_tab3, example_tab4, example_tab5 = st.tabs(
    [
        "TF-IDF Output", #done
        "POS + TF-IDF Output", # done
        "POS Keyword Output",
        "Zero-Shot Output",
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
    st.write("This output was too generic and noisy, capturing many common words that don't convey the unique vibe of the listing. We saw the strongest results with the bigrams apprograch, but it still included irrelevant phrases and missed key vibe elements.")

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
    st.write("TDIF did not work even with POS filtering -- because all these listings are so similar. TF-IDF is great for recognizing unique words in a dataset quickly, however we are looking to extract keywords that might be common for some cities. For the city in Flordia we picked, we are hoping to pull 'Ocean View' as a vibe keyword -- TF-IDF won't value 'Ocean View' if most listings contain those words.")

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
    st.write("The method uses a rule-based NLP pipeline that combines normalization, domain-specific stopword filtering, POS-tag-based adjective–noun phrase extraction, lemmatization, and heuristic semantic pruning to identify high-signal descriptive phrases representing listing ambience or “vibe.” " \
    "The downside is that it may miss some nuanced or less frequent vibe indicators. Additionally it requires setting up large hand-crafted stopword lists and rules, which may need to be adjusted for different markets or listing styles.")

with example_tab4:
    st.markdown("**Predicted vibe labels / classification outputs**")
    st.code(
        """
Zero Shot Keywords / Phrases:
- Rustic Canyon
- Rural Malibu
- Incredible Canyon
- Ocean Views
- Quiet Neighborhood
""",
        language="text",
    )
    st.markdown("**Commentary**")
    st.write("Zero Shot is better for vibe-based keyword extraction because it uses contextual understanding to map text to meaningful, human-defined themes (like “cozy” or “luxury”) rather than relying on surface-level word frequency, resulting in more intuitive and experience-focused keywords.")

with example_tab5:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Final extracted keywords / vibe labels**")
        st.code(
            """
Final Keywords / Vibe Labels:
- Rustic Canyon
- Rural Malibu
- Incredible Canyon
- Ocean Views
- Quiet Neighborhood
""",
            language="text",
        )

    with col2:
        st.markdown("**Emotion scores (NRC EmoLex)**")
        st.code(
            """
PLACEHOLDER:
joy: 0.09
trust: 0.06
anticipation: 0.05
surprise: 0.03
sadness: 0.00
fear: 0.00
anger: 0.00
disgust: 0.00
""",
            language="text",
        )

    st.markdown("**Playlist interpretation**")
    st.write(
        "The NRC Emotion Lexicon works by matching words in a text to a predefined dictionary where each word is associated with specific emotions (like joy, trust, or fear), then counting how frequently those emotion-linked words appear to produce an overall emotional profile.")

st.divider()


# -----------------------------Final Pipeline
section_header(
    "Final NLP Pipeline",
    "The current workflow combines semantic keyword extraction with emotion scoring."
)

st.markdown(
    """
1. **Ingest Airbnb listing description**  
2. **Clean and preprocess text**  
3. **Apply Zero-Shot Text Classification** to identify vibe-oriented labels  
4. **Apply NRC EmoLex** to estimate emotional tone  
5. **Pass keywords + emotions into the Spotify pipeline**    
"""
)

st.success(
    """
**Why this works:** Zero-shot classification provides concept-level vibe labels,
while NRC EmoLex adds emotional nuance. Together, they create a richer representation
of the listing than keyword extraction alone.
"""
)

st.divider()


# ------------------------ Emotion scoring
section_header(
    "Emotion Scoring with NRC EmoLex",
    """
Keywords capture what a listing is about, but emotion scoring helps capture how the listing feels.
This layer adds emotional texture that supports playlist generation.
"""
)

emotion_df = pd.DataFrame(
    {
        "Emotion": [
            "Joy",
            "Trust",
            "Anticipation",
            "Surprise",
            "Sadness",
            "Fear",
            "Anger",
            "Disgust",
        ],
        "Score": [0.35, 0.42, 0.28, 0.10, 0.05, 0.03, 0.02, 0.01],  # placeholder values
    }
)

st.bar_chart(emotion_df.set_index("Emotion"))

st.divider()


# -----------------------------
# Section 6: Why this matters
# -----------------------------
section_header(
    "Why the NLP Layer Matters",
    """
The quality of the playlist depends on the quality of the text interpretation.
Better keyword extraction leads to better emotional inference, which leads to a
playlist that feels more aligned with the stay experience.
"""
)

st.markdown("### End-to-End Example")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Input Text**")
    st.write("Airbnb description")

with col2:
    st.markdown("**Keywords / Vibe**")
    st.write("Extracted semantic labels")

with col3:
    st.markdown("**Emotions**")
    st.write("Emotion profile from NRC EmoLex")

with col4:
    st.markdown("**Playlist Direction**")
    st.write("Spotify recommendation logic")

st.divider()

st.markdown("### Key Takeaways")
st.markdown(
    """
- TF-IDF was useful as a baseline, but not ideal for extracting listing vibe  
- POS filtering improved quality by reducing generic language  
- Simpler extraction methods gave more interpretable results  
- Zero-shot classification best captured semantic atmosphere  
- NRC EmoLex added emotional depth that improved downstream playlist generation  
"""
)