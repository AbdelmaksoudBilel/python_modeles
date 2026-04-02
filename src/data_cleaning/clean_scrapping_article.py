import re
import json
import time
import hashlib
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# CONFIGURATION

OUTPUT_FILE = "data/scraped_articles.json"
DELAY       = 1.5      # secondes entre chaque requête (respecter les serveurs)
TIMEOUT     = 15       # timeout par requête

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
}


# LISTE DES URLS AVEC MÉTADONNÉES MANUELLES
# trouble    : "TSA" | "RM" | "TSA+RM"
# categorie  : catégorie principale (sera aussi auto-détectée)
# source_nom : nom lisible de la source

URLS = [

    # ── TSA : Soutenir les aidants ────────────────────────────────────────
    {
        "url"        : "https://www.soutenirlesaidants.fr/articles/mediation-aidants-aides-faciliter-la-relation-entre-l-aidant-et-le-proche-aide",
        "trouble"    : "TSA+RM",
        "categorie"  : "general",
        "source_nom" : "Soutenir les Aidants",
    },
    {
        "url"        : "https://www.soutenirlesaidants.fr/articles/aidants-des-conseils-pour-prendre-soin-de-vous",
        "trouble"    : "TSA+RM",
        "categorie"  : "general",
        "source_nom" : "Soutenir les Aidants",
    },

    # ── TSA : Autisme Info Service ────────────────────────────────────────
    {
        "url"        : "https://www.autismeinfoservice.fr/adapter/professionnels-sante/reagir",
        "trouble"    : "TSA",
        "categorie"  : "crise",
        "source_nom" : "Autisme Info Service",
    },
    {
        "url"        : "https://www.autismeinfoservice.fr/adapter/professionnels-education/education-structuree",
        "trouble"    : "TSA",
        "categorie"  : "scolarite",
        "source_nom" : "Autisme Info Service",
    },
    {
        "url"        : "https://www.autismeinfoservice.fr/accompagner/travailler-enfants-autistes/teacch",
        "trouble"    : "TSA",
        "categorie"  : "scolarite",
        "source_nom" : "Autisme Info Service – TEACCH",
    },

    # ── TSA : Maison de l'Autisme ─────────────────────────────────────────
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/se-former-proche-aidant-autisme/",
        "trouble"    : "TSA",
        "categorie"  : "general",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/proche-aidant-personne-autiste/",
        "trouble"    : "TSA",
        "categorie"  : "general",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/harcelement-scolaire-chez-les-eleves-avec-un-tnd/",
        "trouble"    : "TSA",
        "categorie"  : "scolarite",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/college-autisme-tnd/",
        "trouble"    : "TSA",
        "categorie"  : "scolarite",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/douleur-personne-autiste-tnd/",
        "trouble"    : "TSA",
        "categorie"  : "sensoriel",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/relations-sociales-autisme/",
        "trouble"    : "TSA",
        "categorie"  : "social",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/particularites-alimentaires-autisme/",
        "trouble"    : "TSA",
        "categorie"  : "alimentation",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/se-deplacer-quand-on-est-autiste/",
        "trouble"    : "TSA",
        "categorie"  : "autonomie",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/loisirs-autisme/",
        "trouble"    : "TSA",
        "categorie"  : "social",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/vacances-autisme/",
        "trouble"    : "TSA",
        "categorie"  : "social",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/autisme-et-tdah/",
        "trouble"    : "TSA",
        "categorie"  : "comportement",
        "source_nom" : "Maison de l'Autisme",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/fiches-pratiques-autisme/",
        "trouble"    : "TSA",
        "categorie"  : "general",
        "source_nom" : "Maison de l'Autisme – Index fiches",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/glossaire-autisme/",
        "trouble"    : "TSA",
        "categorie"  : "general",
        "source_nom" : "Maison de l'Autisme – Glossaire",
    },
    {
        "url"        : "https://maisondelautisme.gouv.fr/accueil-se-former-tnd/etp-autisme/",
        "trouble"    : "TSA",
        "categorie"  : "general",
        "source_nom" : "Maison de l'Autisme – ETP",
    },

    # ── TSA : Comprendre l'Autisme ────────────────────────────────────────
    {
        "url"        : "https://comprendrelautisme.com/lautisme/les-pathologies-associees/",
        "trouble"    : "TSA",
        "categorie"  : "general",
        "source_nom" : "Comprendre l'Autisme",
    },

    # ── TSA : MSD Manuals ─────────────────────────────────────────────────
    {
        "url"        : "https://www.msdmanuals.com/fr/accueil/probl%C3%A8mes-de-sant%C3%A9-infantiles/troubles-de-l-apprentissage-et-du-d%C3%A9veloppement/trouble-du-spectre-autistique",
        "trouble"    : "TSA",
        "categorie"  : "general",
        "source_nom" : "MSD Manuals",
    },

    # ── RM : Québec Éducation ─────────────────────────────────────────────
    {
        "url"        : "https://www.quebec.ca/education/prescolaire-primaire-et-secondaire/ressources-outils-reseau-scolaire/eleves-handicapes-difficultes-adaptation-apprentissage/programmes-eleves-deficience-moyenne-severe",
        "trouble"    : "RM",
        "categorie"  : "scolarite",
        "source_nom" : "Gouvernement du Québec",
    },

    # ── RM : INSERM ───────────────────────────────────────────────────────
    {
        "url"        : "https://ipubli.inserm.fr/bitstream/handle/10608/6816/Chapitre_1.html",
        "trouble"    : "RM",
        "categorie"  : "general",
        "source_nom" : "INSERM",
    },

    # ── RM : ScienceDirect ────────────────────────────────────────────────
    {
        "url"        : "https://www.sciencedirect.com/science/article/pii/S187506722030016X",
        "trouble"    : "RM",
        "categorie"  : "autonomie",
        "source_nom" : "ScienceDirect",
    },

    # ── RM : Vie de Parents ───────────────────────────────────────────────
    {
        "url"        : "https://www.viedeparents.ca/rendre-nos-enfants-autonomes-et-responsables-est-ce-possible/",
        "trouble"    : "RM",
        "categorie"  : "autonomie",
        "source_nom" : "Vie de Parents",
    },

    # ── RM : Cairn ────────────────────────────────────────────────────────
    {
        "url"        : "https://shs.cairn.info/revue-empan-2016-4-page-31",
        "trouble"    : "RM",
        "categorie"  : "general",
        "source_nom" : "Cairn – Empan",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION DU CONTENU HTML
# ─────────────────────────────────────────────────────────────────────────────

# Sélecteurs CSS par domaine  (pour cibler le contenu utile)
SELECTORS = {
    "maisondelautisme.gouv.fr" : "main, article, .entry-content, .page-content",
    "autismeinfoservice.fr"    : "main, article, .content, #main-content",
    "soutenirlesaidants.fr"    : "main, article, .article-body, .content",
    "comprendrelautisme.com"   : "main, article, .entry-content",
    "msdmanuals.com"           : "article, .content, #topicContent",
    "quebec.ca"                : "main, article, #contenu",
    "ipubli.inserm.fr"         : "body",
    "sciencedirect.com"        : "article, .article-content, section",
    "viedeparents.ca"          : "main, article, .entry-content",
    "cairn.info"               : "article, .texte, .article-body",
    "default"                  : "main, article, .content, #content, body",
}

# Balises à supprimer dans tous les cas
TAGS_TO_REMOVE = [
    'nav', 'footer', 'header', 'aside', 'script', 'style',
    'noscript', 'form', 'button', 'figure', 'figcaption',
    '.menu', '.sidebar', '.breadcrumb', '.cookie', '.newsletter',
    '.share', '.social', '.related', '.advertisement',
]


def get_selector(url: str) -> str:
    """Retourne le sélecteur CSS adapté au domaine."""
    for domain, selector in SELECTORS.items():
        if domain in url:
            return selector
    return SELECTORS["default"]


def scrape_url(url: str) -> str | None:
    """
    Scrape une URL et retourne le texte nettoyé.

    Returns:
        Texte extrait ou None si échec
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding  # détection encodage
    except requests.exceptions.SSLError:
        # Retry sans vérification SSL (certains sites gouvernementaux)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                                verify=False)
            resp.encoding = resp.apparent_encoding
        except Exception as e:
            return None
    except Exception as e:
        return None

    soup = BeautifulSoup(resp.text, 'html.parser')

    # ── Supprimer les balises inutiles ────────────────────────────────────
    for tag_name in TAGS_TO_REMOVE:
        if tag_name.startswith('.'):
            for el in soup.select(tag_name):
                el.decompose()
        else:
            for el in soup.find_all(tag_name):
                el.decompose()

    # ── Cibler le contenu principal ───────────────────────────────────────
    selector = get_selector(url)
    content  = soup.select_one(selector)

    if not content:
        content = soup.find('body')

    if not content:
        return None

    # ── Extraire le texte ─────────────────────────────────────────────────
    text = content.get_text(separator='\n', strip=True)
    return text if len(text.strip()) > 100 else None


# NETTOYAGE DU TEXTE WEB

def clean_web_text(text: str) -> str:
    """Nettoie le texte extrait d'une page web."""

    # Supprimer les lignes trop courtes (menus, boutons, breadcrumbs...)
    lines = text.split('\n')
    lines = [l.strip() for l in lines if len(l.strip().split()) >= 4
             or len(l.strip()) == 0]
    text  = '\n'.join(lines)

    # Supprimer répétitions de caractères spéciaux (---|===|...)
    text = re.sub(r'[-=_*]{4,}', '', text)

    # Supprimer URLs seules sur une ligne
    text = re.sub(r'^\s*https?://\S+\s*$', '', text, flags=re.MULTILINE)

    # Supprimer les lignes qui ressemblent à du JS/CSS résiduel
    text = re.sub(r'^\s*[{};]\s*$', '', text, flags=re.MULTILINE)

    # Normaliser les sauts de ligne
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normaliser les espaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Supprimer caractères de contrôle
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text.strip()


def detect_language(text: str) -> str:
    """Détection simple fr / en."""
    sample   = text[:500].lower()
    fr_score = sum(1 for w in ['le ','la ','les ','un ','une ','est ','pour ','dans '] if w in sample)
    en_score = sum(1 for w in ['the ','is ','are ','with ','for ','in ','of ','and '] if w in sample)
    return "fr" if fr_score >= en_score else "en"


# PIPELINE PRINCIPAL

def run():
    Path("data").mkdir(exist_ok=True)

    # Dédupliquer les URLs (la même URL apparaît 2x dans la liste)
    seen_urls = set()
    unique_urls = []
    for item in URLS:
        url_clean = item["url"].split("#")[0].split("?")[0].rstrip("/")
        if url_clean not in seen_urls:
            seen_urls.add(url_clean)
            unique_urls.append(item)

    print(f"\n{'='*60}")
    print(f"  SCRAPING DES ARTICLES WEB")
    print(f"  URLs uniques : {len(unique_urls)}")
    print(f"{'='*60}\n")

    all_articles = []
    success      = 0
    errors       = []

    for i, item in enumerate(unique_urls, 1):
        url        = item["url"]
        source_nom = item["source_nom"]
        trouble    = item["trouble"]
        categorie  = item["categorie"]

        print(f"[{i:02d}/{len(unique_urls)}] {source_nom}")
        print(f"         {url[:70]}...")

        # ── Scraping ──────────────────────────────────────────────────────
        raw_text = scrape_url(url)

        if not raw_text:
            print(f"         ✘ Échec (accès refusé ou site indisponible)\n")
            errors.append({'url': url, 'source': source_nom})
            time.sleep(DELAY)
            continue

        # ── Nettoyage ─────────────────────────────────────────────────────
        text       = clean_web_text(raw_text)
        word_count = len(text.split())
        langue     = detect_language(text)

        if word_count < 50:
            print(f"         ✘ Texte trop court ({word_count} mots)\n")
            errors.append({'url': url, 'source': source_nom, 'raison': 'trop court'})
            time.sleep(DELAY)
            continue

        print(f"         ✔ {word_count:,} mots | langue={langue}")
        apercu = ' '.join(text.split()[:15])
        print(f"         Aperçu : {apercu}...\n")

        # ── Stocker ───────────────────────────────────────────────────────
        doc_id = hashlib.md5(url.encode()).hexdigest()[:8]

        all_articles.append({
            'doc_id'     : doc_id,
            'source_nom' : source_nom,
            'source_url' : url,
            'source_type': 'site_officiel',
            'trouble'    : trouble,
            'categorie'  : categorie,
            'langue'     : langue,
            'word_count' : word_count,
            'text'       : text,
        })

        success += 1
        time.sleep(DELAY)   # pause entre requêtes

    # ── Sauvegarde ────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    # ── Résumé ────────────────────────────────────────────────────────────
    total_mots = sum(a['word_count'] for a in all_articles)
    tsa_count  = sum(1 for a in all_articles if a['trouble'] == 'TSA')
    rm_count   = sum(1 for a in all_articles if a['trouble'] == 'RM')
    both_count = sum(1 for a in all_articles if a['trouble'] == 'TSA+RM')

    print(f"{'='*60}")
    print(f"  RÉSUMÉ FINAL")
    print(f"{'='*60}")
    print(f"  Succès         : {success}/{len(unique_urls)}")
    print(f"  Mots total     : {total_mots:,}")
    print(f"  Articles TSA   : {tsa_count}")
    print(f"  Articles RM    : {rm_count}")
    print(f"  Articles TSA+RM: {both_count}")

    if errors:
        print(f"\n  ✘ Échecs ({len(errors)}) :")
        for e in errors:
            print(f"     - {e['source']} : {e['url'][:60]}")

    print(f"\n  Sauvegardé → {OUTPUT_FILE}")
    print(f"{'='*60}")
    print(f"\n✔ Prêt pour l'étape suivante : chunking")

    return all_articles


if __name__ == "__main__":
    run()