![Chat Interface](Chat.png)

# Idea 3: AI-Powered Document Search and Summarization System

---

Vidéo : https://drive.google.com/file/d/1mkb1AZ4KO26LMFO7Z3ah_JePeyYqZ3yv/view?usp=sharing

Slides : https://drive.google.com/file/d/1PZxTaPSPz-uNpfhZp-ogQQ85qZr2Ys6x/view?usp=drive_link

---

# 🎯 Objectif global

Ce projet implémente un système complet de **Recherche Augmentée par les Documents (RAG)** optimisé pour fonctionner efficacement **sur CPU**.
Il permet :

* d’ingérer des documents (.txt, .pdf, .docx),
* d’effectuer une **recherche sémantique** avec FAISS,
* de produire automatiquement un **résumé** des passages les plus pertinents.

---

# 🧩 Étape 1 — Définir le périmètre du projet

### Workflow global

1. Définir le périmètre
2. Importer & extraire le texte
3. Découper en chunks
4. Générer les embeddings
5. Construire l’index vectoriel
6. Créer l’interface de recherche
7. Résumer les passages pertinents
8. Évaluer qualité & performance
9. Automatiser l’ingestion
10. Documenter architecture & résultats

### Formats supportés

* `.txt`
* `.pdf`
* `.docx`

---

# 📥 Étape 2 — Importation & extraction du texte

### Interface

* **Streamlit** → upload via `st.file_uploader()`

### Extraction

* TXT → Python natif
* PDF → PyPDF2 ou PyMuPDF (fitz)
* DOCX → python-docx

### Sous-échantillonnage (optionnel)

* slicing ou `random.sample`

---

# ✂️ Étape 3 — Découper le texte en chunks

### Chunking

* tailles de 250 tokens
* comptage avec `AutoTokenizer`

### Nettoyage

* suppression headers/footers via `re`

---

# 🧠 Étape 4 — Génération des embeddings

### Modèle

* `sentence-transformers/all-MiniLM-L6-v2`

### Calcul

* `encode()` avec `batch_size=2 ou batch_size=4` adapté CPU

---

# 🗃️ Étape 5 — Construction de l’index vectoriel

### FAISS (CPU)

* `IndexFlatL2`
* recherche via `.search(query, k)`

### Métadonnées

* stockage du chunk + docID

---

# 🔍 Étape 6 — Recherche sémantique

1. Embedding de la requête
2. Recherche k-NN (FAISS)
3. Retour des *top-k* chunks

---

# 📝 Étape 7 — Résumé automatique

### Modèles disponibles

* `facebook/bart-base` (qualité supérieure)
* `t5-small` (léger)

### Pipeline

* `pipeline("summarization", ...)`

### Traitement

* concaténation des chunks avant résumé

---

# 📊 Étape 8 — Évaluation du système

### Recherche

* precision@k
* recall@k
* via Python ou `scikit-learn`

### Résumé

* métriques `bleu`, `rouge`, `perplexity`
* via la librairie `evaluate`

---

# 🤖 Étape 9 — Automatisation (optionnel)

Outils possibles :

* `watchdog` (surveillance dossier)
* `schedule` (tâches récurrentes)
* `cron` (planification système)
* `Celery + Redis` (pipeline async)

---

# 📜 Étape 10 — Documentation & analyse

Inclure :

* architecture RAG
* choix techniques
* performances CPU
* limites & améliorations possibles

---

# 📂 Arborescence du projet

```
PSTB-DI-Bootcamp_Hackathon_2/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ app.py                       # Interface Streamlit
├─ utils.py                     # Extraction, chunks, embeddings, FAISS, résumé
├─ data/
│  ├─ raw/                      # Documents bruts
│  ├─ processed/                # Chunks nettoyés
│  ├─ embeddings/               # Embeddings sauvegardés
│  └─ index/                    # Index FAISS
```

---

# 🚀 Lancer l’application

### 1. Cloner le projet

```
git clone https://github.com/user257814938/AI-Powered-Document-Search-and-Summarization-System
cd AI-Powered-Document-Search-and-Summarization-System
```

### 2. Créer l’environnement

```
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
```

### 3. Installer les dépendances

```
pip install -r requirements.txt
```

### 4. Lancer l’app Streamlit

```
python -m streamlit run app.py
```

Puis :

* upload de documents
* saisie requête
* affichage des passages pertinents
* résumé automatique
