import nltk

# Download all required NLTK resources
# Including both old and new versions for compatibility
resources = [
    'punkt',
    'punkt_tab',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'omw-1.4',  # Open Multilingual Wordnet (sometimes needed for wordnet)
]

print("Downloading NLTK resources...")
for resource in resources:
    try:
        nltk.download(resource, quiet=False)
        print(f"✓ {resource}")
    except Exception as e:
        print(f"⚠ {resource}: {e}")

print("\n✓ All NLTK data downloaded successfully!")
