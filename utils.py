# Objectif — Implémenter le backend RAG : extraction → chunking → embeddings → index FAISS → recherche → résumé → évaluation → persistance

# Étape 1 — Importer les fonctionnalités futures (annotations)
from __future__ import annotations                                              # from : activer les annotations futures | __future__ : module spécial | import annotations : différer l’évaluation des types

# Étape 2 — Importer les bibliothèques standards
import json                                                                     # import : charger un module | json : lire/écrire du JSON
from dataclasses import dataclass                                               # from : importer depuis un module | dataclasses : utilitaires pour classes | import dataclass : décorateur de dataclass
from pathlib import Path                                                        # from : importer depuis un module | pathlib : chemins orientés-objet | import Path : classe chemin
from typing import List, Sequence                                               # from : importer depuis un module | typing : types Python | import List, Sequence : types génériques

# Étape 3 — Importer les bibliothèques ML / NLP
import faiss                                                                    # import : charger un module | faiss : index vectoriel pour recherche de similarité
import numpy as np                                                              # import : charger un module | numpy : calcul numérique | as np : alias
from docx import Document                                                       # from : importer depuis un module | docx : lecture Word | import Document : classe Document
from PyPDF2 import PdfReader                                                    # from : importer depuis un module | PyPDF2 : lecture PDF | import PdfReader : lecteur PDF
from sentence_transformers import SentenceTransformer                           # from : importer depuis un module | sentence_transformers : embeddings NLP | import SentenceTransformer : modèle d’embeddings
from transformers import AutoTokenizer, pipeline                                # from : importer depuis un module | transformers : modèles HF | import AutoTokenizer, pipeline : tokenizer auto + pipeline


# -------------------------
# Data structures
# -------------------------

# Objectif — Définir la structure de données pour stocker les chunks indexés

# Étape 4 — Créer une dataclass IndexedChunk
@dataclass                                                                      # @dataclass : générer __init__/repr automatiquement
class IndexedChunk:                                                             # class : définir une classe | IndexedChunk : nom de structure
    text: str                                                                   # text : contenu textuel du chunk | str : type chaîne
    doc_id: str                                                                 # doc_id : identifiant du document source | str : type chaîne
    chunk_id: int                                                               # chunk_id : identifiant du chunk dans le doc | int : type entier


# -------------------------
# Extraction
# -------------------------

# Objectif — Extraire le texte brut depuis .txt / .pdf / .docx
    
# Étape 5 — Définir la fonction extract_text
def extract_text(file_path: Path) -> str:                                       # def : définir une fonction | extract_text : nom | file_path : chemin fichier | -> str : texte extrait
    suffix = file_path.suffix.lower()                                           # suffix : extension fichier | = : affectation | file_path.suffix : extension | .lower() : normaliser
    if suffix == ".txt":                                                        # if : condition | ".txt" : cas fichier texte
        return file_path.read_text(encoding="utf-8", errors="ignore")           # return : renvoyer texte | read_text : lecture | encoding/errors : robustesse
    if suffix == ".pdf":                                                        # if : condition | ".pdf" : cas fichier PDF
        reader = PdfReader(str(file_path))                                      # reader : lecteur PDF | = : affectation | PdfReader(...) : ouvrir PDF
        return "\n".join(page.extract_text() or "" for page in reader.pages)    # return : concat pages | extract_text() : texte page | or "" : fallback
    if suffix == ".docx":                                                       # if : condition | ".docx" : cas fichier Word
        doc = Document(str(file_path))                                          # doc : document Word | = : affectation | Document(...) : ouvrir docx
        return "\n".join(p.text for p in doc.paragraphs)                        # return : concat paragraphes | p.text : texte paragraphe
    raise ValueError(f"Extension non supportée: {suffix}")                      # raise : lever une erreur si extension inconnue


# -------------------------
# Chunking
# -------------------------

# Objectif — Construire un tokenizer et découper le texte en chunks tokenisés

# Étape 6 — Définir build_tokenizer
def build_tokenizer(model_name: str = "bert-base-uncased"):                     # def : définir une fonction | build_tokenizer : nom | model_name : id modèle | défaut bert-base-uncased
    return AutoTokenizer.from_pretrained(model_name)                            # return : renvoyer tokenizer HF pré-entraîné

# Étape 7 — Définir chunk_text
def chunk_text(                                                                 # def : définir une fonction | chunk_text : nom
    text: str,                                                                  # text : texte complet | str : type chaîne
    tokenizer=None,                                                             # tokenizer : tokenizer optionnel | None : par défaut
    chunk_size: int = 250,                                                      # chunk_size : taille chunk en tokens | défaut 250
    overlap: int = 30                                                           # overlap : recouvrement en tokens | défaut 30
) -> List[str]:                                                                 # -> List[str] : liste de segments texte
    tokenizer = tokenizer or build_tokenizer()                                  # tokenizer : utiliser celui fourni sinon le construire
    tokens = tokenizer.encode(text, add_special_tokens=False)                   # tokens : ids du texte | add_special_tokens=False : sans CLS/SEP
    chunks = []                                                                 # chunks : liste de sortie
    step = max(chunk_size - overlap, 1)                                         # step : pas de glissement | max(...,1) : éviter un pas nul

    for start in range(0, len(tokens), step):                                   # for : boucle de découpage par fenêtres glissantes
        piece = tokens[start : start + chunk_size]                              # piece : slice de tokens de taille chunk_size
        if not piece:                                                           # if : chunk vide
            continue                                                            # continue : sauter
        decoded = tokenizer.decode(piece, skip_special_tokens=True)             # decoded : texte chunk décodé
        if decoded.strip():                                                     # if : chunk non vide après nettoyage
            chunks.append(decoded.strip())                                      # append : ajouter chunk

    return chunks                                                               # return : renvoyer la liste de chunks


# -------------------------
# Embeddings
# -------------------------

# Objectif — Construire un embedder SBERT et encoder les chunks en vecteurs
    
# Étape 8 — Définir build_embedder
def build_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"): # def : définir une fonction | build_embedder : nom | modèle embeddings par défaut
    return SentenceTransformer(model_name, device="cpu")                        # return : renvoyer le modèle sur CPU

# Étape 9 — Définir encode_chunks
def encode_chunks(                                                              # def : définir une fonction | encode_chunks : nom
    chunks: Sequence[str],                                                      # chunks : séquence de textes | Sequence[str] : type
    embedder=None,                                                              # embedder : encodeur optionnel
    batch_size: int = 4                                                         # batch_size : taille des lots
) -> np.ndarray:                                                                # -> np.ndarray : matrice embeddings
    embedder = embedder or build_embedder()                                     # embedder : utiliser fourni sinon construire
    embeddings = embedder.encode(                                               # embeddings : calcul embeddings
        list(chunks),                                                           # list(chunks) : convertir en liste
        batch_size=batch_size,                                                  # batch_size : batching CPU
        normalize_embeddings=True,                                              # normalize_embeddings=True : normaliser les vecteurs
        convert_to_numpy=True,                                                  # convert_to_numpy=True : retour NumPy
        show_progress_bar=False,                                                # show_progress_bar=False : pas de barre
    )
    return embeddings.astype("float32")                                         # return : forcer dtype float32 compatible FAISS


# -------------------------
# Index
# -------------------------

# Objectif — Construire, sauvegarder et recharger un index FAISS L2
    
# Étape 10 — Définir build_faiss_index
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:             # def : définir une fonction | build_faiss_index : nom
    if embeddings.ndim != 2:                                                    # if : vérifier que embeddings est 2D
        raise ValueError("Les embeddings doivent être de forme (n, dim)")       # raise : erreur si mauvaise forme
    dim = embeddings.shape[1]                                                   # dim : dimension des vecteurs
    index = faiss.IndexFlatL2(dim)                                              # index : index FAISS distance L2
    index.add(embeddings)                                                       # index.add : ajouter les vecteurs à l’index
    return index                                                                # return : renvoyer l’index

# Étape 11 — Définir save_index
def save_index(index: faiss.IndexFlatL2, path: Path) -> None:                   # def : définir une fonction | save_index : nom
    path.parent.mkdir(parents=True, exist_ok=True)                              # mkdir : créer le dossier de sortie
    faiss.write_index(index, str(path))                                         # write_index : écrire l’index sur disque

# Étape 12 — Définir load_index
def load_index(path: Path) -> faiss.IndexFlatL2:                                # def : définir une fonction | load_index : nom
    return faiss.read_index(str(path))                                          # return : lire l’index depuis disque


# -------------------------
# Recherche
# -------------------------

# Objectif — Effectuer une recherche top-k dans l’index FAISS
    
# Étape 13 — Définir search
def search(embeddings: np.ndarray, index: faiss.IndexFlatL2, top_k: int = 5):   # def : définir une fonction | search : nom
    scores, idxs = index.search(embeddings, top_k)                              # scores, idxs : distances L2 et indices récupérés
    return scores, idxs                                                         # return : renvoyer scores + indices


# -------------------------
# Résumé
# -------------------------

# Objectif — Construire un pipeline de résumé et résumer les chunks récupérés

# Étape 14 — Définir build_summarizer
def build_summarizer(model_name: str = "t5-small"):                             # def : définir une fonction | build_summarizer : nom
    return pipeline(                                                            # return : créer pipeline HF
        "summarization",                                                        # "summarization" : tâche
        model=model_name,                                                       # model=model_name : identifiant modèle
        device=-1,                                                              # device=-1 : CPU
        truncation=True,                                                        # truncation=True : tronquer l’entrée si trop longue
    )

# Étape 15 — Définir summarize_chunks
def summarize_chunks(                                                           # def : définir une fonction | summarize_chunks : nom
    chunks: Sequence[IndexedChunk],                                             # chunks : séquence d’IndexedChunk
    summarizer=None,                                                            # summarizer : pipeline optionnel
    max_length: int = 200,                                                      # max_length : longueur max du résumé
    min_length: int = 30                                                        # min_length : longueur min du résumé
) -> str:                                                                       # -> str : résumé final
    summarizer = summarizer or build_summarizer()                               # summarizer : utiliser fourni sinon construire
    merged = "\n".join(chunk.text for chunk in chunks)                          # merged : fusion des chunks en un texte
    summary = summarizer(                                                       # summary : appel pipeline
        merged,                                                                 # merged : texte à résumer
        max_length=max_length,                                                  # max_length : contrainte max
        min_length=min_length,                                                  # min_length : contrainte min
        no_repeat_ngram_size=3,                                                 # no_repeat_ngram_size=3 : limiter répétitions
        truncation=True,                                                        # truncation=True : tronquer entrée si besoin
    )
    return summary[0]["summary_text"].strip()                                   # return : extraire le résumé texte


# -------------------------
# Évaluation (optionnelle)
# -------------------------

# Objectif — Calculer precision@k et recall@k sur des ids pertinents

# Étape 16 — Définir precision_recall_at_k
def precision_recall_at_k(                                                      # def : définir une fonction | precision_recall_at_k : nom
    relevant_ids: Sequence[int],                                                # relevant_ids : ids pertinents
    retrieved_ids: Sequence[int],                                               # retrieved_ids : ids récupérés
    k: int = 5                                                                  # k : top-k
):
    relevant_set = set(relevant_ids)                                            # relevant_set : ensemble des ids pertinents
    retrieved_top_k = retrieved_ids[:k]                                         # retrieved_top_k : k premiers ids récupérés
    true_positive = len([rid for rid in retrieved_top_k if rid in relevant_set])# true_positive : nb d’ids pertinents dans le top-k
    precision = true_positive / max(k, 1)                                       # precision : proportion de pertinents dans le top-k
    recall = true_positive / max(len(relevant_set), 1)                          # recall : proportion de pertinents retrouvés
    return precision, recall                                                    # return : renvoyer (precision, recall)


# -------------------------
# Persist metadata
# -------------------------

# Objectif — Sauvegarder et recharger les métadonnées des chunks en JSON

# Étape 17 — Définir save_metadata
def save_metadata(chunks: Sequence[IndexedChunk], path: Path) -> None:          # def : définir une fonction | save_metadata : nom
    path.parent.mkdir(parents=True, exist_ok=True)                              # mkdir : créer le dossier parent
    with path.open("w", encoding="utf-8") as f:                                 # with : ouvrir un fichier JSON en écriture
        json.dump(                                                              # json.dump : écrire JSON
            [chunk.__dict__ for chunk in chunks],                               # [chunk.__dict__ ...] : sérialiser les chunks en dicts
            f,                                                                  # f : handle fichier
            ensure_ascii=False,                                                 # ensure_ascii=False : conserver accents
            indent=2,                                                           # indent=2 : format lisible
        )

# Étape 18 — Définir load_metadata
def load_metadata(path: Path) -> List[IndexedChunk]:                            # def : définir une fonction | load_metadata : nom
    data = json.loads(path.read_text(encoding="utf-8"))                         # data : contenu JSON parsé
    return [IndexedChunk(**item) for item in data]                              # return : reconstruire les IndexedChunk
