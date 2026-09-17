import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from natural_lingam import NLPOrderCausalDiscovery

def main(): 
    # 1. Load real dataset
    reviews_df = pd.read_csv('cleaned_makeup_reviews.csv')
    products_df = pd.read_csv('cleaned_makeup_products.csv')

    raw_product_texts = products_df['best_uses'].dropna().astype(str).tolist()
    raw_texts = reviews_df['comments'].dropna().astype(str).tolist()

    # 2. Extract Top Terms (Unigrams + Bigrams) by Mean TF-IDF
    #    ngram_range=(1, 2) enables both single words and 2-word phrases
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),        # Includes unigrams and bigrams
        stop_words='english',
        min_df=5,
        token_pattern=r'(?u)\b[\w_]+\b' # Ensures compounds/bigrams preserve characters
    )
    
    # Fit initial TF-IDF matrix across full corpus
    X_full = tfidf_vectorizer.fit_transform(raw_product_texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Rank terms by average TF-IDF score
    mean_tfidf = np.asarray(X_full.mean(axis=0)).ravel()
    top_100_indices = mean_tfidf.argsort()[::-1][:100]
    vocab = [feature_names[i] for i in top_100_indices]

    # Re-transform corpus using ONLY the selected vocabulary (Unigrams + Bigrams)
    final_vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b[\w_]+\b'
    )
    X_dtm = final_vectorizer.fit_transform(raw_product_texts).toarray().astype(np.float64)

    print(f"Top TF-IDF Vocabulary (including Bigrams) selected:\n{vocab}\n")

    # 3. Initialize and fit causal discovery pipeline
    model = NLPOrderCausalDiscovery(
        spacy_model="en_core_web_sm",
        cv=5,
        n_jobs=-1
    )
    
    model.fit(raw_product_texts, X_dtm, vocab)

    # 4. Display Results
    ordered_vocab = [vocab[i] for i in model.pi]
    print("Inferred Topological Order (π):")
    print(" -> ".join(ordered_vocab) + "\n")

    # Extract non-zero causal edges (w_i -> w_j)
    edges = []
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            weight = model.W_[i, j]
            if abs(weight) > 1e-4:
                edges.append((vocab[i], vocab[j], weight))

    # Sort edges by absolute causal weight strength
    edges.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"Found {len(edges)} causal edges. Top edges:")
    for src, tgt, w in edges:
        print(f"   {src} ---> {tgt}  (weight: {w:.4f})")

if __name__ == "__main__":
    main()