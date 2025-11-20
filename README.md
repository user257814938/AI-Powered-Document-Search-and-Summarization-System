# PSTB-DI-Bootcamp_Hackathon_2
Idea 3: AI-Powered Document Search and Summarization System






Il devrait inclure :

Nom du projet
Définition, problème résolu et caractéristiques
Comment l'exécuter (sur l'ordinateur de quelqu'un d'autre)





1. Choix des Composants

Framework Web : Streamlit (Rapidité de dév).

Ingestion : PyPDF2 & python-docx (Légers et compatibles CPU).

Embeddings : sentence-transformers/all-MiniLM-L6-v2.

Pourquoi ? Rapide, faible empreinte mémoire, dimension 384 idéale pour CPU.

Vector Store : FAISS (IndexFlatL2).

Pourquoi ? Standard industriel, version CPU très performante pour <100k vecteurs.

Summarization : facebook/bart-base.

Pourquoi ? Meilleure cohérence que T5-small, bien que légèrement plus lent. Fenêtre de contexte de 1024 tokens.
