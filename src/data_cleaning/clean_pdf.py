import os, re, sys, json
from pathlib import Path

try:
    from deep_translator import PdfReader
    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False
    print("pip install pypdf")

try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_OK = True
except ImportError:
    OCR_OK = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
PDF_DIR            = "data/rag"
OUTPUT_FILE        = "data/extracted_pages.json"
MIN_WORDS_FOR_TEXT = 10        # seuil abaissé (était 30)
OCR_LANG           = "fra+eng"
OCR_DPI            = 200

# ── DIAGNOSTIC ────────────────────────────────────────────────────────────────
def diagnose_pdf(pdf_path):
    result = {"status": "ok", "pages": 0, "avg_words": 0, "message": ""}
    if not PYPDF_OK:
        result.update({"status": "error", "message": "pypdf non installé"})
        return result
    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except:
                result.update({"status": "encrypted",
                               "message": "PDF protégé par mot de passe"})
                return result
        nb = len(reader.pages)
        result["pages"] = nb
        if nb == 0:
            result.update({"status": "empty", "message": "PDF sans pages"})
            return result
        total = 0
        sample = min(5, nb)
        for i in range(sample):
            try:
                total += len((reader.pages[i].extract_text() or "").split())
            except:
                pass
        avg = total / sample
        result["avg_words"] = avg
        if avg < MIN_WORDS_FOR_TEXT:
            result.update({"status": "image",
                           "message": f"PDF scanné (moy {avg:.1f} mots/page) → OCR"})
        else:
            result.update({"status": "ok",
                           "message": f"PDF numérique OK (moy {avg:.1f} mots/page)"})
    except Exception as e:
        result.update({"status": "corrupted",
                       "message": f"PDF corrompu : {type(e).__name__}: {e}"})
    return result

# ── EXTRACTION TEXTE ──────────────────────────────────────────────────────────
def extract_text_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        reader.decrypt("")
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except:
            text = ""
        if len(text.strip()) >= 30:
            pages.append({"page": i+1, "text": text, "method": "text"})
    return pages

# ── EXTRACTION OCR ────────────────────────────────────────────────────────────
def extract_ocr_pdf(pdf_path):
    if not OCR_OK:
        raise ImportError("pdf2image/pytesseract non installés")
    print(f"     Conversion → images (dpi={OCR_DPI})...")
    images = convert_from_path(pdf_path, dpi=OCR_DPI)
    print(f"     {len(images)} pages")
    pages = []
    for i, img in enumerate(images):
        print(f"     OCR {i+1}/{len(images)}...", end="\r", flush=True)
        try:
            text = pytesseract.image_to_string(img, lang=OCR_LANG)
        except:
            text = ""
        if len(text.strip()) >= 30:
            pages.append({"page": i+1, "text": text, "method": "ocr"})
    print()
    return pages

# ── EXTRACTION MIXTE ──────────────────────────────────────────────────────────
def extract_mixed_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        reader.decrypt("")
    pages, ocr_needed = [], []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except:
            text = ""
        if len(text.split()) >= MIN_WORDS_FOR_TEXT:
            pages.append({"page": i+1, "text": text, "method": "text"})
        else:
            ocr_needed.append(i+1)
    if ocr_needed and OCR_OK:
        print(f"     {len(ocr_needed)} pages vides → OCR...")
        for pnum in ocr_needed:
            imgs = convert_from_path(pdf_path, dpi=OCR_DPI,
                                     first_page=pnum, last_page=pnum)
            if imgs:
                try:
                    text = pytesseract.image_to_string(imgs[0], lang=OCR_LANG)
                    if len(text.split()) >= MIN_WORDS_FOR_TEXT:
                        pages.append({"page": pnum, "text": text, "method": "ocr"})
                except:
                    pass
    pages.sort(key=lambda p: p["page"])
    return pages

# ── NETTOYAGE ─────────────────────────────────────────────────────────────────
def clean_text(text):
    for old, new in [("ﬁ","fi"),("ﬂ","fl"),("ﬀ","ff"),("ﬃ","ffi"),
                     ("\x0c","\n"),("\xa0"," ")]:
        text = text.replace(old, new)
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    lines = [l for l in text.split("\n")
             if len(l.strip().split()) >= 4 or not l.strip()]
    text  = "\n".join(lines)
    text  = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text  = re.sub(r"\n{3,}", "\n\n", text)
    text  = re.sub(r"[ \t]{2,}", " ", text)
    text  = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()

def detect_language(text):
    s  = text[:500].lower()
    fr = sum(1 for w in ["le ","la ","les ","un ","une ","est ","pour ","dans "] if w in s)
    en = sum(1 for w in ["the ","is ","are ","with ","for ","in ","of ","and "] if w in s)
    return "fr" if fr >= en else "en"

# ── TRAITEMENT D'UN PDF ───────────────────────────────────────────────────────
def process_pdf(pdf_path):
    diag = diagnose_pdf(pdf_path)
    print(f"   Diagnostic        : {diag['message']}")

    # Causes d'échec irrémédiables
    if diag["status"] == "encrypted":
        print(f"   ✘ SOLUTION : Ouvrir le PDF → Enregistrer sous (sans mot de passe)")
        return None
    if diag["status"] == "corrupted":
        print(f"   ✘ SOLUTION : Ré-télécharger ou réparer avec Adobe Acrobat")
        return None
    if diag["status"] == "empty":
        print(f"   ✘ SOLUTION : Vérifier le fichier dans un lecteur PDF")
        return None

    # Extraction
    raw_pages = []
    if diag["status"] == "image":
        if not OCR_OK:
            print(f"   ✘ SOLUTION : pip install pdf2image pytesseract pillow")
            print(f"              + sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng")
            return None
        try:
            raw_pages = extract_ocr_pdf(pdf_path)
        except Exception as e:
            print(f"   ✘ Erreur OCR : {e}")
            return None
    else:
        try:
            raw_pages = extract_text_pdf(pdf_path)
        except Exception as e:
            print(f"   ✘ Erreur extraction : {e}")
            return None
        # Fallback mixte si résultat insuffisant
        total_wc = sum(len(p["text"].split()) for p in raw_pages)
        if total_wc < 100 and diag["pages"] > 1:
            print(f"   ⚠️  Texte insuffisant ({total_wc} mots) → fallback mixte...")
            try:
                raw_pages = extract_mixed_pdf(pdf_path)
            except Exception as e:
                print(f"   ✘ Erreur fallback : {e}")
                if not raw_pages:
                    return None

    print(f"   Pages extraites   : {len(raw_pages)}")
    if not raw_pages:
        print(f"   ✘ Aucune page extraite")
        return None

    # Nettoyage
    clean_pages = []
    for p in raw_pages:
        cleaned = clean_text(p["text"])
        wc = len(cleaned.split())
        if wc < 20:
            continue
        clean_pages.append({
            "page"      : p["page"],
            "text"      : cleaned,
            "word_count": wc,
            "method"    : p["method"],
            "langue"    : detect_language(cleaned),
        })

    if not clean_pages:
        print(f"   ✘ Tout vide après nettoyage")
        return None

    langues    = [p["langue"] for p in clean_pages]
    langue_doc = max(set(langues), key=langues.count)
    total_mots = sum(p["word_count"] for p in clean_pages)
    ocr_count  = sum(1 for p in clean_pages if p["method"] == "ocr")

    print(f"   Pages conservées  : {len(clean_pages)}")
    print(f"   Mots total        : {total_mots:,}")
    print(f"   Langue            : {langue_doc}")
    if ocr_count:
        print(f"   Pages OCR         : {ocr_count}")
    print(f"   Aperçu            : {' '.join(clean_pages[0]['text'].split()[:20])}...")

    return {
        "fichier"    : Path(pdf_path).name,
        "source_nom" : Path(pdf_path).stem,
        "langue"     : langue_doc,
        "pdf_type"   : diag["status"],
        "total_pages": len(clean_pages),
        "total_mots" : total_mots,
        "pages"      : clean_pages,
    }

# ── TRAITEMENT FICHIER TXT ───────────────────────────────────────────────────
def process_txt(txt_path):

    print(f"   Lecture fichier TXT")

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except:
        try:
            with open(txt_path, "r", encoding="latin-1") as f:
                text = f.read()
        except Exception as e:
            print(f"   ✘ Erreur lecture TXT : {e}")
            return None

    cleaned = clean_text(text)

    wc = len(cleaned.split())

    if wc < 20:
        print("   ✘ TXT trop court")
        return None

    langue_doc = detect_language(cleaned)

    pages = [{
        "page": 1,
        "text": cleaned,
        "word_count": wc,
        "method": "txt",
        "langue": langue_doc
    }]

    print(f"   Mots total        : {wc:,}")
    print(f"   Langue            : {langue_doc}")
    print(f"   Aperçu            : {' '.join(cleaned.split()[:20])}...")

    return {
        "fichier": Path(txt_path).name,
        "source_nom": Path(txt_path).stem,
        "langue": langue_doc,
        "pdf_type": "txt",
        "total_pages": 1,
        "total_mots": wc,
        "pages": pages
    }


# ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────
def run():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except:
            pass

    os.makedirs("data", exist_ok=True)
    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
    txt_files = sorted(Path(PDF_DIR).glob("*.txt"))

    all_files = pdf_files + txt_files

    if not pdf_files:
        print(f"\n⚠️  Aucun PDF trouvé dans : {PDF_DIR}")
        return

    print(f"\n{'='*62}")
    print(f"  EXTRACTION PDFs  (v2 corrigée)")
    print(f"  Dossier  : {PDF_DIR}")
    print(f"  PDFs     : {len(pdf_files)}")
    print(f"  TXT      : {len(txt_files)}")
    print(f"  Total    : {len(all_files)}")
    print(f"  OCR      : {'✔ disponible' if OCR_OK else '✘ non disponible'}")
    print(f"{'='*62}\n")

    all_docs, errors = [], []

    for file_path in all_files:
        print(f"📄 {file_path.name}")
        if file_path.suffix.lower() == ".pdf":
            doc = process_pdf(str(file_path))
        else:
            doc = process_txt(str(file_path))
        if doc:
            all_docs.append(doc)
        else:
            errors.append(file_path.name)
        print()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    total_pages = sum(d["total_pages"] for d in all_docs)
    total_mots  = sum(d["total_mots"]  for d in all_docs)

    print(f"{'='*62}")
    print(f"  RÉSUMÉ FINAL")
    print(f"{'='*62}")
    print(f"  Documents traités : {len(all_docs)}/{len(all_files)}")
    print(f"  Pages conservées  : {total_pages:,}")
    print(f"  Mots total        : {total_mots:,}")
    print(f"  Français          : {sum(1 for d in all_docs if d['langue']=='fr')} doc(s)")
    print(f"  Anglais           : {sum(1 for d in all_docs if d['langue']=='en')} doc(s)")
    print(f"  Traités par OCR   : {sum(1 for d in all_docs if d['pdf_type']=='image')} doc(s)")

    if errors:
        print(f"\n  ✘ Échecs ({len(errors)}) :")
        for e in errors:
            print(f"     - {e}")
        print(f"\n  Causes possibles et solutions :")
        print(f"     🔒 PDF protégé  → Ouvrir + Enregistrer sous (sans mot de passe)")
        print(f"     📷 PDF scanné   → Installer tesseract (OCR)")
        print(f"     💥 PDF corrompu → Ré-télécharger le fichier")
        print(f"     📄 PDF vide     → Vérifier dans un lecteur PDF")

    print(f"\n  Sauvegardé → {OUTPUT_FILE}")
    print(f"{'='*62}")

if __name__ == "__main__":
    run()