import numpy as np
import spacy
from sklearn.linear_model import ElasticNetCV
from joblib import Parallel, delayed
from tqdm import tqdm

class NLPOrderCausalDiscovery:
    """
    Scalable Causal Discovery (DAG Estimation) using SVO Semantic Priors.
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
        
        self.connective_verbs = set(connective_verbs or {
            "cause", "lead", "trigger", "result", "induce", 
            "produce", "bring", "create", "drive", "prompt"
        })

    def _extract_svo_triples(self, doc):
        edges = []
        for token in doc:
            if token.pos_ in ["VERB", "AUX"]:
                subjects = [c for c in token.children if "subj" in c.dep_]
                objects = [c for c in token.children if "obj" in c.dep_ or "attr" in c.dep_]
                
                for subj in subjects:
                    subj_lemma = subj.lemma_.lower()
                    
                    for obj in objects:
                        obj_lemma = obj.lemma_.lower()
                        if subj_lemma != obj_lemma:
                            edges.append((subj_lemma, obj_lemma))
                            
                    for prep in [c for c in token.children if c.dep_ == "prep"]:
                        for pobj in [c for c in prep.children if "obj" in c.dep_]:
                            pobj_lemma = pobj.lemma_.lower()
                            if subj_lemma != pobj_lemma:
                                edges.append((subj_lemma, pobj_lemma))
                                
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

        filtered_texts = []
        for text in tqdm(raw_texts, desc="Filtering relevant texts"):
            text_lower = text.lower()
            hits = sum(1 for word in vocab_set if word in text_lower)
            if hits >= 2:
                filtered_texts.append(text)

        parsed_docs = self.nlp.pipe(filtered_texts, batch_size=1000)
        
        for doc in tqdm(parsed_docs, total=len(filtered_texts), desc="Parsing SpaCy SVO trees"):
            edges = self._extract_svo_triples(doc)
            for src, tgt in edges:
                if src in word2idx and tgt in word2idx:
                    i, j = word2idx[src], word2idx[tgt]
                    precedence_matrix[i, j] += 1.0

        net_asymmetry = precedence_matrix.sum(axis=1) - precedence_matrix.sum(axis=0)
        
        for verb in self.connective_verbs:
            if verb in word2idx:
                net_asymmetry[word2idx[verb]] -= 1e6
                
        return np.argsort(-net_asymmetry)

    def _is_self_directed(self, source_vocab, target_vocab):
        """
        Check if source and target share token overlapping definitions.
        Example: 'dark' -> 'dark circles' or 'sensitive skin' -> 'skin'
        """
        src_tokens = set(source_vocab.lower().split())
        tgt_tokens = set(target_vocab.lower().split())
        return src_tokens.issubset(tgt_tokens) or tgt_tokens.issubset(src_tokens)

    def _fit_single_variable(self, k, X_permuted, vocab):
        if k == 0:
            return np.zeros(0)
            
        target_vocab = vocab[self.pi[k]]
        y_target = X_permuted[:, k]
        
        if np.std(y_target) == 0:
            return np.zeros(k)

        # Identify antecedent indices in topological order that are NOT self-directed
        valid_antecedent_indices = []
        for source_perm_idx in range(k):
            source_vocab = vocab[self.pi[source_perm_idx]]
            if not self._is_self_directed(source_vocab, target_vocab):
                valid_antecedent_indices.append(source_perm_idx)

        # If all antecedents overlap with target, return zero array of size k
        if not valid_antecedent_indices:
            return np.zeros(k)

        X_antecedents = X_permuted[:, valid_antecedent_indices]

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
            cv=self.cv, 
            positive=True, 
            random_state=self.random_state, 
            max_iter=3000
        )
        model.fit(X_antecedents, y_target)
        
        # Map fitted coefficients back to full k-length antecedent array
        full_coefs = np.zeros(k)
        for idx, orig_k_idx in enumerate(valid_antecedent_indices):
            full_coefs[orig_k_idx] = model.coef_[idx]

        return full_coefs

    def fit(self, raw_texts, X_dtm, vocab):
        d = len(vocab)
        vocab = np.array(vocab)
        
        # 1. Topological Ordering via SVO Path Extraction
        self.pi = self._compute_topological_order(raw_texts, vocab)
        
        # Permute columns according to topological order pi
        X_permuted = X_dtm[:, self.pi]

        # 2. Parallel Edge Selection (Independent ElasticNet with Masking)
        print("Fitting parallel ElasticNet regressions across vocabulary...")
        coef_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_variable)(k, X_permuted, vocab) 
            for k in tqdm(range(d), desc="Fitting DAG regressions")
        )

        # Construct strictly lower-triangular matrix
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