import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from natural_lingam import NLPOrderCausalDiscovery

def main(): 
    # 1. Load real dataset
    reviews_df = pd.read_csv('cleaned_makeup_reviews.csv')
    raw_texts = reviews_df['comments'].dropna().astype(str).tolist()

    # 2. Extract Top 10 Words by Mean TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=5
    )
    
    # Fit initial TF-IDF matrix across full corpus
    X_full = tfidf_vectorizer.fit_transform(raw_texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Rank terms by average TF-IDF score
    mean_tfidf = np.asarray(X_full.mean(axis=0)).ravel()
    top_10_indices = mean_tfidf.argsort()[::-1][:100]
    vocab = [feature_names[i] for i in top_10_indices]

    # Re-transform corpus using ONLY the top 10 TF-IDF vocabulary
    final_vectorizer = TfidfVectorizer(vocabulary=vocab)
    X_dtm = final_vectorizer.fit_transform(raw_texts).toarray().astype(np.float64)

    print(f"Top 10 TF-IDF Vocabulary selected:\n{vocab}\n")

    # 3. Initialize and fit causal discovery pipeline
    model = NLPOrderCausalDiscovery(
        spacy_model="en_core_web_sm",
        cv=5,
        n_jobs=-1
    )
    
    model.fit(raw_texts, X_dtm, vocab)

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
        print(f"  {src} ---> {tgt}  (weight: {w:.4f})")

if __name__ == "__main__":
    main()