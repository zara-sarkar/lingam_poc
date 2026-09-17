import numpy as np
import spacy
from sklearn.linear_model import ElasticNetCV
from joblib import Parallel, delayed
from tqdm import tqdm

class NLPOrderCausalDiscovery:
    """
    Scalable Causal Discovery (DAG Estimation) using SVO Semantic Priors.
    
    1. Extracts Subject-Verb-Object paths from dependency trees, bypassing relational verbs 
       (e.g., 'rain' causes 'flooding' -> direct edge 'rain' -> 'flooding').
    2. Ranks vocabulary via net outgoing flow to derive topological order pi.
    3. Fits parallel ElasticNet regressions predicting each variable solely from its antecedents.
    """
    def __init__(
        self, 
        spacy_model="en_core_web_sm", 
        connective_verbs=None, 
        cv=3, 
        n_jobs=-1, 
        random_state=42
    ):
        self.nlp = spacy.load(spacy_model, disable=["ner", "attribute_ruler"])
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.pi = None
        self.W_ = None  # Adjacency matrix (W[i, j] represents edge i -> j)
        
        # Relational connectives to bypass in causal flow calculations
        self.connective_verbs = set(connective_verbs or {
            "cause", "lead", "trigger", "result", "induce", 
            "produce", "bring", "create", "drive", "prompt"
        })

    def _extract_svo_triples(self, doc):
        """Extracts direct (Subject -> Object) precedence edges from dependency trees."""
        edges = []
        for token in doc:
            # Locate active/passive verbs or noun heads
            if token.pos_ in ["VERB", "AUX"]:
                subjects = [c for c in token.children if "subj" in c.dep_]
                objects = [c for c in token.children if "obj" in c.dep_ or "attr" in c.dep_]
                
                # Direct Subject -> Object path (bypasses connective verb)
                for subj in subjects:
                    subj_lemma = subj.lemma_.lower()
                    
                    # Connect to direct objects
                    for obj in objects:
                        obj_lemma = obj.lemma_.lower()
                        if subj_lemma != obj_lemma:
                            edges.append((subj_lemma, obj_lemma))
                            
                    # Handle prepositional objects (e.g., "leads to flooding")
                    for prep in [c for c in token.children if c.dep_ == "prep"]:
                        for pobj in [c for c in prep.children if "obj" in c.dep_]:
                            pobj_lemma = pobj.lemma_.lower()
                            if subj_lemma != pobj_lemma:
                                edges.append((subj_lemma, pobj_lemma))
                                
            # Direct Noun-to-Noun dependencies (e.g., "rain damage")
            elif token.dep_ in ["compound", "nmod"]:
                head_lemma = token.head.lemma_.lower()
                child_lemma = token.lemma_.lower()
                if head_lemma != child_lemma:
                    edges.append((child_lemma, head_lemma))
                    
        return edges

    def _compute_topological_order(self, raw_texts, vocab):
        d = len(vocab)
        vocab_set = set(word.lower() for word in vocab)
        word2idx = {word.lower(): i for i, word in enumerate(vocab)}
        precedence_matrix = np.zeros((d, d), dtype=np.float64)

        # 1. PRE-FILTER: Keep only reviews containing AT LEAST TWO vocabulary words
        filtered_texts = []
        for text in tqdm(raw_texts, desc="Filtering relevant texts"):
            text_lower = text.lower()
            hits = sum(1 for word in vocab_set if word in text_lower)
            if hits >= 2:
                filtered_texts.append(text)

        print(f"Pre-filtered {len(raw_texts)} reviews down to {len(filtered_texts)} relevant reviews for SpaCy parsing.")

        # 2. Fast SpaCy processing with tqdm progress bar
        parsed_docs = self.nlp.pipe(filtered_texts, batch_size=1000)
        
        for doc in tqdm(parsed_docs, total=len(filtered_texts), desc="Parsing SpaCy SVO trees"):
            edges = self._extract_svo_triples(doc)
            for src, tgt in edges:
                if src in word2idx and tgt in word2idx:
                    i, j = word2idx[src], word2idx[tgt]
                    precedence_matrix[i, j] += 1.0

        # 3. Compute Net Flow Score
        net_asymmetry = precedence_matrix.sum(axis=1) - precedence_matrix.sum(axis=0)
        
        for verb in self.connective_verbs:
            if verb in word2idx:
                net_asymmetry[word2idx[verb]] -= 1e6
                
        return np.argsort(-net_asymmetry)

    def _fit_single_variable(self, k, X_permuted):
        if k == 0:
            return np.zeros(0)
            
        X_antecedents = X_permuted[:, :k]
        y_target = X_permuted[:, k]
        
        if np.std(y_target) == 0:
            return np.zeros(k)

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
            cv=self.cv, 
            positive=True, 
            random_state=self.random_state, 
            max_iter=3000
        )
        model.fit(X_antecedents, y_target)
        return model.coef_

    def fit(self, raw_texts, X_dtm, vocab):
        d = len(vocab)
        
        # 1. Topological Ordering via SVO Path Extraction
        self.pi = self._compute_topological_order(raw_texts, vocab)
        
        # Permute columns according to topological order pi
        X_permuted = X_dtm[:, self.pi]

        # 2. Parallel Edge Selection (Independent ElasticNet) with tqdm progress bar
        print("Fitting parallel ElasticNet regressions across vocabulary...")
        coef_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_variable)(k, X_permuted) 
            for k in tqdm(range(d), desc="Fitting DAG regressions")
        )

        # Construct strictly lower-triangular matrix (Acyclicity guaranteed)
        W_permuted = np.zeros((d, d), dtype=np.float64)
        for k in range(1, d):
            W_permuted[k, :k] = coef_list[k]

        # Map back to original vocabulary order: W_[i, j] represents w_i -> w_j
        self.W_ = np.zeros((d, d), dtype=np.float64)
        for target_idx_perm in range(d):
            target_idx_orig = self.pi[target_idx_perm]
            for source_idx_perm in range(target_idx_perm):
                source_idx_orig = self.pi[source_idx_perm]
                self.W_[source_idx_orig, target_idx_orig] = W_permuted[target_idx_perm, source_idx_perm]

        return self