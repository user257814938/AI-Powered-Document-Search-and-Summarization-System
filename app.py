# Objectif — Construire une app Streamlit RAG CPU-friendly (upload → chunking → embeddings → FAISS → recherche → résumé)

# Étape 1 — Importer les bibliothèques et fonctions utilitaires
import tempfile                                                                   # import : charger un module | tempfile : gérer des dossiers temporaires
from pathlib import Path                                                          # from : importer depuis un module | pathlib : gestion des chemins | import Path : classe chemin
from typing import List                                                           # from : importer depuis un module | typing : types Python | import List : liste typée
import streamlit as st                                                            # import : charger un module | streamlit : UI web Python | as st : alias local

from utils import (                                                               # from : importer depuis un module | utils : fichier utilitaire local | import (...) : fonctions/classes utilisées
    IndexedChunk,                                                                 # IndexedChunk : dataclass (texte + ids)
    build_embedder,                                                               # build_embedder : construire l’encodeur d’embeddings
    build_faiss_index,                                                            # build_faiss_index : créer un index FAISS
    build_summarizer,                                                             # build_summarizer : construire le modèle de résumé
    build_tokenizer,                                                              # build_tokenizer : construire le tokenizer
    chunk_text,                                                                   # chunk_text : découper le texte en chunks
    encode_chunks,                                                                # encode_chunks : encoder chaque chunk en embedding
    extract_text,                                                                 # extract_text : extraire le texte depuis un fichier
    search,                                                                       # search : chercher dans FAISS
    summarize_chunks,                                                             # summarize_chunks : résumer une liste de chunks
)

# Étape 2 — Définir un dossier cache local
def get_cache_dir() -> Path:                                                      # def : définir une fonction | get_cache_dir : nom | -> Path : retourne un chemin
    base = Path(tempfile.gettempdir()) / "rag_streamlit"                          # base : dossier cache | = : affectation | Path(tempfile.gettempdir()) : dossier temporaire système | / "rag_streamlit" : sous-dossier
    base.mkdir(parents=True, exist_ok=True)                                       # base.mkdir : créer le dossier | parents=True : créer les parents | exist_ok=True : ignorer si existe
    return base                                                                   # return : renvoyer le chemin cache

# Étape 3 — Mettre en cache les ressources lourdes (tokenizer, embedder, summarizer)
@st.cache_resource(show_spinner=False)                                           # st.cache_resource : cache persistant Streamlit | show_spinner=False : pas de spinner auto
def get_tokenizer():                                                             # def : définir une fonction | get_tokenizer : nom
    return build_tokenizer()                                                     # return : renvoyer | build_tokenizer() : instance tokenizer

@st.cache_resource(show_spinner=False)                                           # st.cache_resource : cache persistant Streamlit
def get_embedder():                                                              # def : définir une fonction | get_embedder : nom
    return build_embedder()                                                      # return : renvoyer | build_embedder() : instance embedder

@st.cache_resource(show_spinner=False)                                           # st.cache_resource : cache persistant Streamlit
def get_summarizer():                                                            # def : définir une fonction | get_summarizer : nom
    return build_summarizer()                                                    # return : renvoyer | build_summarizer() : instance summarizer

# Étape 4 — Configurer la page Streamlit et le layout
st.set_page_config(page_title="Recherche + Résumé (CPU)", layout="wide")         # st.set_page_config : configurer la page | page_title : titre onglet | layout="wide" : mode large
st.title("🔎 Recherche sémantique + résumé (CPU-friendly)")                      # st.title : titre principal | "🔎 ..." : texte affiché

# Étape 5 — Construire la sidebar (upload + hyperparamètres)
st.sidebar.header("📂 Upload & Préparation")                                     # st.sidebar.header : titre sidebar | "📂 ..." : texte
uploaded_file = st.sidebar.file_uploader(                                        # uploaded_file : fichier uploadé | = : affectation | st.sidebar.file_uploader : widget upload
    "Choisir un fichier",                                                        # "Choisir un fichier" : label du widget
    type=["txt", "pdf", "docx"],                                                 # type=[...] : extensions autorisées
)
chunk_size = st.sidebar.slider(                                                  # chunk_size : taille chunk | = : affectation | st.sidebar.slider : widget slider
    "Taille des chunks (tokens)",                                                # label : texte du slider
    min_value=100,                                                               # min_value : valeur min
    max_value=400,                                                               # max_value : valeur max
    value=250,                                                                   # value : valeur par défaut
    step=25,                                                                     # step : pas
)
overlap = st.sidebar.slider(                                                     # overlap : recouvrement | = : affectation | slider : widget
    "Overlap (tokens)",                                                          # label
    min_value=0,                                                                 # min_value : min
    max_value=100,                                                               # max_value : max
    value=30,                                                                    # value : défaut
    step=10,                                                                     # step : pas
)
batch_size = st.sidebar.select_slider(                                           # batch_size : taille batch embeddings | = : affectation | select_slider : widget
    "Batch embeddings",                                                          # label
    options=[2, 4, 8],                                                           # options : valeurs possibles
    value=4,                                                                     # value : défaut
)
top_k = st.sidebar.slider(                                                       # top_k : nb résultats | = : affectation | slider : widget
    "Top-k résultats",                                                           # label
    min_value=1,                                                                 # min_value : min
    max_value=10,                                                                # max_value : max
    value=5,                                                                     # value : défaut
)

# Étape 6 — Initialiser l’état Streamlit (index, chunks, embeddings)
if "index" not in st.session_state:                                              # if : condition | "index" not in st.session_state : test d’initialisation
    st.session_state.index = None                                                # st.session_state.index : état index FAISS | = None : pas encore construit
    st.session_state.chunks: List[IndexedChunk] = []                             # st.session_state.chunks : liste typée | = [] : vide
    st.session_state.embeddings = None                                           # st.session_state.embeddings : embeddings doc | = None : non calculés

# Étape 7 — Définir la routine d’upload + indexation
def handle_upload():                                                             # def : définir une fonction | handle_upload : callback sidebar
    if not uploaded_file:                                                        # if : condition | not uploaded_file : aucun fichier uploadé
        st.warning("Uploadez un fichier pour démarrer.")                         # st.warning : alerte utilisateur | "..." : message
        return                                                                   # return : sortir de la fonction

    cache_dir = get_cache_dir()                                                  # cache_dir : dossier cache | = : affectation | get_cache_dir() : appel
    dest_path = cache_dir / uploaded_file.name                                   # dest_path : chemin fichier cache | = : affectation | cache_dir / name : concaténation Path
    with dest_path.open("wb") as f:                                              # with : contexte fichier | dest_path.open("wb") : ouvrir en écriture binaire
        f.write(uploaded_file.getbuffer())                                       # f.write : écrire bytes | uploaded_file.getbuffer() : contenu uploadé

    with st.spinner("Extraction du texte..."):                                   # with : spinner UI | "Extraction..." : message spinner
        text = extract_text(dest_path)                                           # text : texte extrait | = : affectation | extract_text(dest_path) : extraction

    tokenizer = get_tokenizer()                                                  # tokenizer : tokenizer caché | = : affectation | get_tokenizer() : appel cache
    with st.spinner("Découpage en chunks..."):                                   # with : spinner UI | "Découpage..." : message
        chunks_text = chunk_text(                                                # chunks_text : liste de chunks | = : affectation | chunk_text(...) : découpe
            text,                                                                # text : texte source
            tokenizer=tokenizer,                                                 # tokenizer=tokenizer : tokenizer utilisé
            chunk_size=chunk_size,                                               # chunk_size=chunk_size : taille choisie via slider
            overlap=overlap,                                                     # overlap=overlap : recouvrement choisi
        )

    if not chunks_text:                                                          # if : condition | not chunks_text : liste vide
        st.error("Aucun texte détecté après découpe.")                           # st.error : message erreur
        return                                                                   # return : sortir

    embedder = get_embedder()                                                    # embedder : modèle embeddings | = : affectation | get_embedder() : appel cache
    with st.spinner("Calcul des embeddings (CPU)..."):                           # with : spinner UI | "Calcul..." : message
        embeddings = encode_chunks(                                              # embeddings : matrice embeddings | = : affectation | encode_chunks(...) : encodage
            chunks_text,                                                         # chunks_text : segments texte
            embedder=embedder,                                                   # embedder=embedder : encodeur
            batch_size=batch_size,                                               # batch_size=batch_size : taille batch
        )

    with st.spinner("Construction de l'index FAISS..."):                         # with : spinner UI | "Construction..." : message
        index = build_faiss_index(embeddings)                                    # index : FAISS index | = : affectation | build_faiss_index(embeddings) : construction

    st.session_state.index = index                                               # st.session_state.index : stocker index | = : affectation
    st.session_state.embeddings = embeddings                                     # st.session_state.embeddings : stocker embeddings | = : affectation
    st.session_state.chunks = [                                                  # st.session_state.chunks : stocker chunks enrichis | = : affectation | [...] : list comprehension
        IndexedChunk(text=chunk, doc_id=uploaded_file.name, chunk_id=i)          # IndexedChunk(...) : construire un item | text : contenu | doc_id : nom fichier | chunk_id : id chunk
        for i, chunk in enumerate(chunks_text)                                   # for : boucle comprehension | enumerate(chunks_text) : (index, chunk)
    ]
    st.success(f"Index construit avec {len(chunks_text)} chunks.")               # st.success : message succès | f"...{len(...) }..." : nb chunks

# Étape 8 — Bouton sidebar pour lancer l’indexation
st.sidebar.button("Indexer le document", on_click=handle_upload)                 # st.sidebar.button : bouton | "Indexer..." : label | on_click=handle_upload : callback

# Étape 9 — Zone de requête utilisateur
st.subheader("Requête")                                                          # st.subheader : sous-titre section | "Requête" : texte
query = st.text_input("Texte de la requête")                                     # query : texte requête | = : affectation | st.text_input : champ input

# Étape 10 — Lancer la recherche et le résumé
if st.button("Lancer la recherche"):                                                        # if : condition | st.button(...) : bouton principal
    if not st.session_state.index:                                                          # if : condition | not index : aucun index
        st.error("Aucun index n'est disponible. Uploadez et indexez un document d'abord.")  # st.error : message erreur
    elif not query.strip():                                                                 # elif : autre condition | not query.strip() : requête vide
        st.warning("La requête est vide.")                                                  # st.warning : avertissement
    else:
        embedder = get_embedder()                                                # embedder : récupérer embedder | = : affectation
        query_emb = embedder.encode(                                             # query_emb : embedding requête | = : affectation | embedder.encode(...) : encodage
            [query],                                                             # [query] : liste d’une requête
            normalize_embeddings=True,                                           # normalize_embeddings=True : normaliser embeddings
            convert_to_numpy=True,                                               # convert_to_numpy=True : sortie NumPy
        ).astype("float32")                                                      # .astype("float32") : convertir dtype float32

        scores, idxs = search(query_emb, st.session_state.index, top_k=top_k)    # scores, idxs : résultats FAISS | = : affectation | search(...) : recherche | top_k=top_k : nb résultats
        best_scores = scores[0]                                                  # best_scores : scores top-k | = : affectation | scores[0] : première requête
        best_idxs = idxs[0]                                                      # best_idxs : indices top-k | = : affectation | idxs[0] : première requête

        retrieved = []                                                           # retrieved : liste résultats | = : affectation | [] : vide
        for score, idx in zip(best_scores, best_idxs):                           # for : boucle | zip(...) : pairs (score, index)
            if idx == -1:                                                        # if : condition | idx == -1 : résultat vide FAISS
                continue                                                         # continue : passer au suivant
            chunk = st.session_state.chunks[idx]                                 # chunk : chunk récupéré | = : affectation | st.session_state.chunks[idx] : accès par index
            retrieved.append((chunk, score))                                     # retrieved.append : ajouter | (chunk, score) : tuple résultat

        if not retrieved:                                                        # if : condition | not retrieved : aucun résultat
            st.info("Aucun résultat retourné.")                                  # st.info : information UI
        else:
            st.markdown("### Résultats")                                         # st.markdown : titre markdown
            for rank, (chunk, score) in enumerate(retrieved, start=1):           # for : boucle | enumerate(..., start=1) : ranking à partir de 1
                st.write(f"**#{rank}** — distance L2: {score:.4f}")              # st.write : afficher texte | f"...{score:.4f}" : score formaté
                st.caption(f"{chunk.doc_id} | chunk {chunk.chunk_id}")           # st.caption : petit texte | doc_id + chunk_id
                st.code(chunk.text)                                              # st.code : afficher code/texte monospacé | chunk.text : contenu

            summarizer = get_summarizer()                                        # summarizer : modèle résumé | = : affectation | get_summarizer() : appel cache
            with st.spinner("Génération du résumé (t5-small, CPU)..."):          # with : spinner UI | "Génération..." : message
                summary = summarize_chunks(                                      # summary : résumé final | = : affectation | summarize_chunks(...) : résumé
                    [item[0] for item in retrieved],                             # [item[0] for item in retrieved] : liste des chunks sans scores
                    summarizer=summarizer,                                       # summarizer=summarizer : modèle de résumé
                )
            st.markdown("### Résumé synthétique")                                # st.markdown : titre markdown
            st.success(summary)                                                  # st.success : afficher résumé dans un bloc succès
