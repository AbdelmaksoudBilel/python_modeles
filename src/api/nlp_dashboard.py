import re, logging
from collections import Counter
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# ── Imports NLP ───────────────────────────────────────────────────────────────
try:
    import spacy
    nlp_fr = spacy.load("fr_core_news_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    logger.warning("spaCy fr non disponible → lemmatisation désactivée")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("scikit-learn non disponible → clustering désactivé")

try:
    from lingua import Language, LanguageDetectorBuilder
    _detector = LanguageDetectorBuilder.from_languages(
        Language.FRENCH, Language.ARABIC, Language.ENGLISH
    ).build()
    LINGUA_OK = True
except ImportError:
    LINGUA_OK = False
    logger.warning("lingua non disponible → détection langue désactivée")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_OK = True
except ImportError:
    TRANSLATOR_OK = False
    logger.warning("deep_translator non disponible → pip install deep-translator")


# STOPWORDS

STOPWORDS_FR = set([
    "mon","ma","mes","le","la","les","de","du","un","une","des","ce","cet",
    "cette","ces","je","il","elle","nous","vous","ils","elles","on","me","te",
    "se","lui","leur","y","en","est","sont","a","ont","être","avoir","faire",
    "que","qui","quoi","comment","pourquoi","quand","où","quel","quelle",
    "et","ou","mais","donc","or","ni","car","pour","par","sur","sous","dans",
    "avec","sans","entre","vers","son","sa","ses","leurs","plus","très","bien",
    "aussi","tout","tous","toute","toutes","pas","ne","si","même","autre",
    "alors","puis","déjà","encore","toujours","jamais","souvent","parfois",
    "veux","peut","dois","faut","va","vais","fait","dit","voir","savoir",
    "enfant","mon","fils","fille","ça","là","ici","comme","quand","après",
    "avant","pendant","depuis","jusqu","bonjour","merci","aide","besoin",
])

STOPWORDS_AR = set([
    "من","إلى","عن","على","في","مع","هو","هي","هم","أنا","أنت","نحن",
    "هذا","هذه","التي","الذي","ما","لا","قد","كان","أن","إن","لم","كل",
    "بعد","قبل","عند","لكن","أو","و","ولكن","إذا","لأن","حتى","ثم",
])

STOPWORDS_EN = set([
    "the","a","an","in","on","at","to","for","of","and","or","but",
    "is","are","was","were","be","have","has","had","do","does","did",
    "my","your","his","her","our","their","it","this","that","these",
    "i","you","he","she","we","they","me","him","us","them","what",
    "how","why","when","where","which","who","not","no","yes","can","will",
])

ALL_STOPWORDS = STOPWORDS_FR | STOPWORDS_AR | STOPWORDS_EN

# Lexique sentiment FR simple
POSITIVE_WORDS = set([
    "progrès","amélioration","mieux","bien","calme","heureux","sourire",
    "réussit","avance","évolution","positif","excellent","super","top",
    "merci","content","satisfait","efficace","fonctionne","marche",
])
NEGATIVE_WORDS = set([
    "crise","agressif","violence","pleure","cri","difficile","problème",
    "impossible","peur","stress","anxieux","fatigue","épuisé","découragé",
    "échec","refuse","blesse","frappe","mord","détruit","fugue","danger",
])


# SCHÉMAS

class NLPRequest(BaseModel):
    messages       : List[str]           # textes bruts des messages parents
    n_keywords     : int = 20            # top N mots-clés
    n_questions    : int = 10            # top N questions
    n_clusters     : int = 5             # nombre de topics
    min_word_length: int = 4             # longueur min des mots


# PIPELINE NLP

class NLPPipeline:

    def __init__(self, min_len: int = 4):
        self.min_len = min_len

    # ── [0] Traduction → FR ───────────────────────────────────────────────────

    def translate_to_fr(self, text: str, lang: str) -> str:
        """Traduit le texte vers le français si ce n'est pas déjà du français."""
        if lang == "fr" or not TRANSLATOR_OK:
            return text
        try:
            translated = GoogleTranslator(source=lang, target="fr").translate(text)
            return translated or text
        except Exception as e:
            logger.warning(f"Traduction échouée ({lang}→fr) : {e}")
            return text

    # ── [1] Nettoyage ─────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)    # ponctuation (garde arabe)
        text = re.sub(r"\d+", " ", text)                      # chiffres
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── [2] Détection langue ──────────────────────────────────────────────────

    def detect_lang(self, text: str) -> str:
        if not LINGUA_OK:
            return "fr"
        try:
            lang = _detector.detect_language_of(text)
            if lang is None: return "fr"
            return {"FRENCH": "fr", "ARABIC": "ar", "ENGLISH": "en"}.get(
                lang.name, "fr"
            )
        except Exception:
            return "fr"

    # ── [3] Tokenisation ──────────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        return [w for w in text.split()
                if len(w) >= self.min_len and w not in ALL_STOPWORDS]

    # ── [4] Segmentation phrases ──────────────────────────────────────────────

    def segment_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"[.!?؟]\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    # ── [5] Lemmatisation (spaCy FR) ──────────────────────────────────────────

    def lemmatize(self, tokens: List[str]) -> List[str]:
        if not SPACY_OK:
            return tokens
        try:
            doc = nlp_fr(" ".join(tokens))
            return [
                token.lemma_ for token in doc
                if token.lemma_ not in ALL_STOPWORDS
                and len(token.lemma_) >= self.min_len
                and not token.is_punct
            ]
        except Exception:
            return tokens

    # ── [6] Détection questions ───────────────────────────────────────────────

    def extract_questions(self, text: str) -> List[str]:
        patterns = [
            r"[Cc]omment\s+.{5,80}[?؟]?",
            r"[Pp]ourquoi\s+.{5,80}[?؟]?",
            r"[Qq]u[ée]\s+.{5,80}[?؟]?",
            r"[Ee]st[-\s]ce\s+.{5,80}[?؟]?",
            r"[Qq]uand\s+.{5,80}[?؟]?",
            r"[Yy]\s+a[-\s]t[-\s]il\s+.{5,80}[?؟]?",
            r"[Cc]'est quoi\s+.{5,80}[?؟]?",
            r".{5,80}[?؟]",
        ]
        questions = []
        for pattern in patterns:
            found = re.findall(pattern, text)
            questions.extend([q.strip() for q in found if len(q.strip()) > 10])
        return questions

    # ── [7] Sentiment ─────────────────────────────────────────────────────────

    def sentiment(self, text: str) -> str:
        words   = set(text.lower().split())
        pos_cnt = len(words & POSITIVE_WORDS)
        neg_cnt = len(words & NEGATIVE_WORDS)
        if neg_cnt > pos_cnt:     return "negative"
        elif pos_cnt > neg_cnt:   return "positive"
        else:                     return "neutral"

    # ── [8] Clustering TF-IDF + KMeans ───────────────────────────────────────

    def cluster_topics(self, texts: List[str], n_clusters: int = 5) -> List[dict]:
        if not SKLEARN_OK or len(texts) < n_clusters:
            return []
        try:
            vectorizer = TfidfVectorizer(
                max_features = 200,
                ngram_range  = (1, 2),
                stop_words   = list(ALL_STOPWORDS),
                min_df       = 2,
            )
            X = vectorizer.fit_transform(texts)
            if X.shape[0] < n_clusters:
                n_clusters = max(2, X.shape[0] // 2)

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            km.fit(X)

            feature_names = vectorizer.get_feature_names_out()
            clusters      = []

            for i in range(n_clusters):
                # Top mots du cluster
                center    = km.cluster_centers_[i]
                top_idx   = center.argsort()[-5:][::-1]
                keywords  = [feature_names[j] for j in top_idx]
                count     = int((km.labels_ == i).sum())

                clusters.append({
                    "topic"   : f"Sujet {i+1}",
                    "keywords": keywords,
                    "count"   : count,
                    "label"   : keywords[0] if keywords else f"Sujet {i+1}",
                })

            return sorted(clusters, key=lambda x: x["count"], reverse=True)

        except Exception as e:
            logger.warning(f"Clustering échoué : {e}")
            return []


# ENDPOINT
@router.post("/dashboard/nlp")
async def nlp_analysis(req: NLPRequest):
    """
    Analyse NLP complète des messages parents.

    Node.js envoie la liste des textes → Python retourne les stats NLP.
    """
    nlp = NLPPipeline(min_len=req.min_word_length)

    all_tokens    = []
    all_questions = []
    sentiments    = {"positive": 0, "neutral": 0, "negative": 0}
    lang_dist     = {"fr": 0, "ar": 0, "en": 0}
    total_length  = 0
    cleaned_texts = []

    for raw in req.messages:
        if not raw or not raw.strip():
            continue

        # Langue d'abord (sur texte original)
        lang = nlp.detect_lang(raw)
        lang_dist[lang] = lang_dist.get(lang, 0) + 1

        # Traduction → FR si nécessaire
        text_fr = nlp.translate_to_fr(raw, lang)

        cleaned      = nlp.clean(text_fr)
        cleaned_texts.append(cleaned)
        total_length += len(cleaned.split())

        # Tokens + lemmes (sur texte FR)
        tokens = nlp.tokenize(cleaned)
        tokens = nlp.lemmatize(tokens)
        all_tokens.extend(tokens)

        # Questions (sur texte FR traduit)
        questions = nlp.extract_questions(text_fr)
        all_questions.extend(questions)

        # Sentiment (sur texte FR)
        sent = nlp.sentiment(cleaned)
        sentiments[sent] += 1

    # ── Fréquences ────────────────────────────────────────────────────────────
    word_freq  = Counter(all_tokens)
    top_keywords = [
        {
            "word" : word,
            "count": count,
            "freq" : round(count / max(len(all_tokens), 1) * 100, 2),
        }
        for word, count in word_freq.most_common(req.n_keywords)
    ]

    # ── Top questions ─────────────────────────────────────────────────────────
    q_counter  = Counter(
        [q.lower().strip().rstrip("?؟").strip() for q in all_questions]
    )
    top_questions = [
        {"question": q, "count": c}
        for q, c in q_counter.most_common(req.n_questions)
        if len(q) > 10
    ]

    # ── Clustering topics ─────────────────────────────────────────────────────
    topic_clusters = nlp.cluster_topics(cleaned_texts, req.n_clusters)

    # ── Word cloud data ───────────────────────────────────────────────────────
    word_cloud = [
        {"text": w, "value": c}
        for w, c in word_freq.most_common(50)
    ]

    # ── Sentiment scores ──────────────────────────────────────────────────────
    total_msgs = max(sum(sentiments.values()), 1)
    sentiment_scores = {
        k: {"count": v, "percent": round(v / total_msgs * 100, 1)}
        for k, v in sentiments.items()
    }

    logger.info(
        f"NLP ✔ | messages={len(req.messages)} | "
        f"tokens={len(all_tokens)} | questions={len(all_questions)}"
    )

    return {
        "top_keywords"    : top_keywords,
        "top_questions"   : top_questions,
        "topic_clusters"  : topic_clusters,
        "sentiment"       : sentiment_scores,
        "word_cloud_data" : word_cloud,
        "avg_msg_length"  : round(total_length / max(len(req.messages), 1), 1),
        "lang_distribution": {
            k: {"count": v, "percent": round(v / max(len(req.messages), 1) * 100, 1)}
            for k, v in lang_dist.items()
        },
        "total_messages_analyzed": len(req.messages),
    }