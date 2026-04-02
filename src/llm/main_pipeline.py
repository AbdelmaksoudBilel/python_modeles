import os, logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Imports modules ────────────────────────────────────────────────────────
from ..multimodal.language_handler import LanguageHandler
from ..multimodal.image_handler    import ImageHandler
from ..multimodal.video_handler    import VideoHandler
from ..multimodal.speech_handler   import SpeechHandler
from ..rag.chunk_filter     import ChunkFilter
from ..rag.memory_manager   import MemoryManager
from ..rag.profile_updater  import ProfileUpdater
from .llm_client       import LLMClient, PromptBuilder


# CONFIGURATION

CHUNKS_FILE = "data/rag/chunk/rag_chunks_meta.json"
FAISS_FILE  = "data/rag/vector/faiss_index.bin"
META_FILE   = "data/rag/vector/metadata.json"

# Mots-clés indiquant un cas grave → orienter vers professionnel
CRITICAL_KEYWORDS = [
    "suicide", "se tuer", "mourir", "mort", "danger de mort",
    "urgence", "hôpital", "crise grave", "convulsion",
]

# Domaines acceptés (domain guard simple)
DOMAIN_KEYWORDS = [
    "autisme", "tsa", "déficience", "handicap", "enfant",
    "comportement", "communication", "thérapie", "école",
    "crise", "sensoriel", "développement", "apprentissage",
    "sommeil", "alimentation", "anxiété", "diagnostic",
]


# PIPELINE PRINCIPAL

class MainPipeline:

    def __init__(self):
        logger.info("Initialisation du pipeline...")

        # LLM
        self.llm     = LLMClient(api_key=os.getenv("GROQ_API_KEY"))
        self.builder = PromptBuilder()

        # Langue
        self.lang_handler = LanguageHandler()

        # Multimodal
        self.image_handler  = ImageHandler(llm_client=None)   # BLIP local
        self.video_handler  = VideoHandler()
        self.speech_handler = SpeechHandler()

        # RAG
        self.chunk_filter   = ChunkFilter(CHUNKS_FILE, FAISS_FILE, META_FILE)

        # Mémoire + Profil
        self.memory_manager  = MemoryManager(llm_client=self.llm)
        self.profile_updater = ProfileUpdater(llm_client=self.llm)

        logger.info("Pipeline prêt ✔")

    # PIPELINE PRINCIPAL

    def run(self,
            question     : str,
            profile      : dict,
            conversation : dict  = None,
            child        : dict  = None,
            media_path   : str   = "",
            media_type   : str   = "") -> dict:
        """
        Exécute le pipeline complet.

        Args:
            question     : message du parent
            profile      : profil enfant depuis la DB (JSON API)
            conversation : { last_5_messages, summary, keywords, total_messages }
            child        : { id, profile_detecter }
            media_path   : chemin vers image/vidéo/audio (optionnel)
            media_type   : "image" | "video" | "audio" | ""

        Returns:
            {
                "answer"          : réponse finale (langue parent),
                "answer_fr"       : réponse en français (interne),
                "parent_lang"     : langue détectée du parent,
                "rag_score"       : score RAG moyen,
                "web_triggered"   : True/False,
                "domain_blocked"  : True/False,
                "critical_alert"  : True/False,
                "updates"         : {
                    "summary"         : nouveau résumé → sauvegarder en DB,
                    "keywords"        : nouveaux mots-clés → sauvegarder en DB,
                    "profile_detecter": profil mis à jour → sauvegarder en DB,
                    "should_update_db": True/False,
                }
            }
        """
        conversation = conversation or {}
        child        = child        or {}
        result       = self._empty_result()

        # ── [1] Traitement multimodal ─────────────────────────────────────
        media_text, media_description = self._process_media(
            media_path, media_type
        )

        # ── [2] Détection langue + traduction FR ──────────────────────────
        full_input   = f"{question} {media_text}".strip()
        lang_result  = self.lang_handler.process(full_input)
        parent_lang  = lang_result["detected_lang"]
        question_fr  = lang_result["translated_text"]
        result["parent_lang"] = parent_lang

        logger.info(f"Langue parent : {parent_lang}")

        # ── [3] Domain Guard ──────────────────────────────────────────────
        if not self._is_in_domain(question_fr):
            logger.warning("Question hors domaine → rejetée")
            result["domain_blocked"] = True
            result["answer"] = self._out_of_domain_response(parent_lang)
            return result

        # ── [4] Alerte cas critique ───────────────────────────────────────
        if self._is_critical(question_fr):
            logger.warning("Cas critique détecté → orientation professionnelle")
            result["critical_alert"] = True
            result["answer"] = self._critical_response(parent_lang)
            return result

        # ── [5] Double recherche RAG + contexte profil ────────────────────
        rag_results = self.chunk_filter.search(
            question = question_fr,
            profile  = profile,
        )
        result["rag_score"]     = rag_results["avg_score"]
        result["web_triggered"] = rag_results["web_triggered"]

        # ── [7] Bloc mémoire ──────────────────────────────────────────────
        memory_block = self.memory_manager.build_memory_block(
            last_5_messages = conversation.get("last_5_messages", []),
            summary         = conversation.get("summary", ""),
            keywords        = conversation.get("keywords", []),
        )

        # ── [8] Construction prompt final ─────────────────────────────────
        messages = self.builder.build(
            question          = question_fr,
            profile_context   = rag_results["profile_context"],
            profile_detecter  = child.get("profile_detecter", []),
            memory_block      = memory_block,
            rag_block         = rag_results["prompt_block"],
            parent_lang       = parent_lang,
            media_description = media_description,
            media_type        = media_type,
        )

        # ── [9] Génération LLM ────────────────────────────────────────────
        answer_fr = self.llm.generate_from_messages(messages)
        result["answer_fr"] = answer_fr

        # ── [10] Traduction réponse → langue parent ───────────────────────
        if parent_lang != "fr":
            answer = self.lang_handler.translate_response_to_parent(
                answer_fr, parent_lang
            )
        else:
            answer = answer_fr
        result["answer"] = answer

        # ── [11] Post-processing : mise à jour mémoire + profil ───────────
        updates = self._post_process(
            conversation = conversation,
            child        = child,
            question_fr  = question_fr,
            answer_fr    = answer_fr,
        )
        result["updates"] = updates

        logger.info("Pipeline terminé ✔")
        return result

    # MODULES INTERNES

    def _process_media(self, media_path: str, media_type: str):
        """
        Traite le média envoyé par le parent.

        Returns:
            (media_text_fr, media_description)
            media_text_fr    : texte extrait traduit en FR (pour question)
            media_description: description courte pour le prompt
        """
        if not media_path or not os.path.exists(media_path):
            return "", ""

        try:
            if media_type == "image":
                r = self.image_handler.process(media_path)
                return r.get("translated_text", ""), r.get("translated_text", "")

            elif media_type == "video":
                r = self.video_handler.process(media_path)
                return r.get("translated_text", ""), r.get("translated_text", "")

            elif media_type == "audio":
                r = self.speech_handler.process(media_path)
                return r.get("translated_text", ""), ""

        except Exception as e:
            logger.warning(f"Erreur traitement média : {e}")

        return "", ""

    def _is_in_domain(self, text: str) -> bool:
        """Domain guard : vérifie si la question est dans le domaine TSA/RM."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in DOMAIN_KEYWORDS)

    def _is_critical(self, text: str) -> bool:
        """Détecte les cas graves nécessitant une orientation professionnelle."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in CRITICAL_KEYWORDS)

    def _post_process(self, conversation: dict, child: dict,
                      question_fr: str, answer_fr: str) -> dict:
        """
        Met à jour mémoire conversationnelle et profil détecté.

        Returns:
            { summary, keywords, profile_detecter, should_update_db }
        """
        total_messages = conversation.get("total_messages", 0) + 2

        # Mise à jour mémoire
        mem_update = self.memory_manager.update_after_response(
            last_5_messages = conversation.get("last_5_messages", []),
            summary         = conversation.get("summary", ""),
            keywords        = conversation.get("keywords", []),
            new_question    = question_fr,
            new_answer      = answer_fr,
            total_messages  = total_messages,
        )

        # Mise à jour profil détecté
        prof_update = self.profile_updater.update(
            profile_detecter = child.get("profile_detecter", []),
            new_question     = question_fr,
            new_answer       = answer_fr,
        )

        return {
            "summary"         : mem_update["summary"],
            "keywords"        : mem_update["keywords"],
            "profile_detecter": prof_update["profile_detecter"],
            "should_update_db": mem_update["should_update_db"] or prof_update["updated"],
            "memory_changes"  : mem_update,
            "profile_changes" : prof_update["changes"],
        }

    def _empty_result(self) -> dict:
        return {
            "answer"         : "",
            "answer_fr"      : "",
            "parent_lang"    : "fr",
            "rag_score"      : 0.0,
            "web_triggered"  : False,
            "domain_blocked" : False,
            "critical_alert" : False,
            "updates"        : {
                "summary"         : "",
                "keywords"        : [],
                "profile_detecter": [],
                "should_update_db": False,
            },
        }

    def _out_of_domain_response(self, lang: str) -> str:
        msgs = {
            "fr": (
                "Je suis spécialisé dans l'accompagnement des parents "
                "d'enfants avec TSA ou déficience intellectuelle. "
                "Pourriez-vous reformuler votre question dans ce contexte ?"
            ),
            "ar": (
                "أنا متخصص في دعم أولياء أمور الأطفال المصابين بالتوحد "
                "أو الإعاقة الذهنية. هل يمكنك إعادة صياغة سؤالك ؟"
            ),
            "en": (
                "I specialize in supporting parents of children with ASD "
                "or intellectual disabilities. "
                "Could you rephrase your question in this context?"
            ),
        }
        return msgs.get(lang, msgs["fr"])

    def _critical_response(self, lang: str) -> str:
        msgs = {
            "fr": (
                "⚠️ La situation que vous décrivez nécessite une aide "
                "professionnelle immédiate. Contactez votre médecin, "
                "un service d'urgence ou appelez le 15 (SAMU)."
            ),
            "ar": (
                "⚠️ الوضع الذي تصفه يتطلب مساعدة متخصصة فورية. "
                "يرجى الاتصال بطبيبك أو خدمات الطوارئ."
            ),
            "en": (
                "⚠️ The situation you describe requires immediate "
                "professional help. Please contact your doctor "
                "or emergency services."
            ),
        }
        return msgs.get(lang, msgs["fr"])


# TEST

if __name__ == "__main__":

    pipeline = MainPipeline()

    result = pipeline.run(
        question = "Comment gérer les crises de mon enfant le matin ?",

        profile = {
            "prediction"        : "TSA",
            "confidence"        : 0.89,
            "Age_Years"         : 5,
            "Sex"               : "M",
            "PR_QF1A"           : 3,
            "PR_QQ"             : 3,
            "PR_QN1_D"          : 1,
            "PR_QO1_A_COMBINE"  : 1,
        },

        conversation = {
            "last_5_messages" : [
                {"role": "user",      "content": "Il crie dès qu'on le réveille."},
                {"role": "assistant", "content": "Les transitions sont difficiles."},
            ],
            "summary"        : "Enfant TSA de 5 ans, crises matinales fréquentes.",
            "keywords"       : ["TSA", "crise", "matin"],
            "total_messages" : 6,
        },

        child = {
            "id"              : "child_123",
            "profile_detecter": ["non verbal", "crises le matin"],
        },
    )

    print("\n" + "="*60)
    print("  RÉSULTAT PIPELINE")
    print("="*60)
    print(f"  Langue parent   : {result['parent_lang']}")
    print(f"  Score RAG       : {result['rag_score']}")
    print(f"  Web déclenché   : {result['web_triggered']}")
    print(f"  Hors domaine    : {result['domain_blocked']}")
    print(f"  Cas critique    : {result['critical_alert']}")
    print(f"\n  RÉPONSE :\n{result['answer']}")
    print(f"\n  MISES À JOUR DB :")
    print(f"    should_update : {result['updates']['should_update_db']}")
    print(f"    profil ajouts : {result['updates']['profile_changes'].get('added', [])}")
    print("="*60)