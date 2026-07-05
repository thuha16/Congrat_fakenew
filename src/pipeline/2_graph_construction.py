import os
import pickle
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def main():
    data_dir = "../data/FakeNewsNet/completed_data"
    
    print("Loading intermediate data...")
    with open(os.path.join(data_dir, "intermediate_data.pkl"), 'rb') as f:
        data = pickle.load(f)
        
    news_ids = data['news_ids']
    news_entities = data['news_entities']
    entities = data['entities']
    corpus = data['corpus']
    lda_model = data['lda_model']
    labels = data['labels']
    
    g = nx.Graph()
    
    # 1. Add Nodes
    print("Adding nodes...")
    # Add News nodes
    for i, n_id in enumerate(news_ids):
        # type='0' is real, '1' is fake based on utils.py
        node_type = '1' if labels[i] == 1 else '0'
        g.add_node(n_id, type=node_type)
        
    # Add Entity nodes
    for ent in entities:
        g.add_node(ent, type='entity')
        
    # Add Topic nodes
    num_topics = 50
    for i in range(num_topics):
        g.add_node(f"topic_{i}", type='topic')
        
    # 2. Add Edges
    print("Adding edges...")
    
    # News-Topic Edges (K=2 topics per news)
    for i, n_id in enumerate(news_ids):
        topics = lda_model.get_document_topics(corpus[i])
        # Sort by probability
        topics = sorted(topics, key=lambda x: x[1], reverse=True)
        top_k = topics[:2]
        for t_id, prob in top_k:
            g.add_edge(n_id, f"topic_{t_id}")
            
    # News-Entity Edges
    for i, n_id in enumerate(news_ids):
        for ent in news_entities[i]:
            g.add_edge(n_id, ent)
            
    # Entity-Entity Edges (Cosine similarity > 0.5)
    # We will build a simple dummy TF-IDF for entities based on their co-occurrence in news
    print("Computing entity similarity for edges...")
    entity_docs = {ent: [] for ent in entities}
    for i, ents in enumerate(news_entities):
        for ent in ents:
            entity_docs[ent].append(news_ids[i])
            
    # Represent each entity as a string of news IDs it appears in
    ent_corpus = [' '.join(entity_docs[ent]) for ent in entities]
    
    if len(entities) > 0:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(ent_corpus)
        
        # To avoid OOM, we can do it in batches or sparse dot product
        # Here we just do it sparse
        similarity = tfidf_matrix * tfidf_matrix.T
        
        # Extract pairs > 0.5
        cx = similarity.tocoo()
        count = 0
        for i, j, v in zip(cx.row, cx.col, cx.data):
            if i < j and v > 0.5:
                g.add_edge(entities[i], entities[j])
                count += 1
        print(f"Added {count} entity-entity edges.")
        
    # 3. Save Graph
    print(f"Graph nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()}")
    with open(os.path.join(data_dir, "model_network_handled.pkl"), 'wb') as f:
        pickle.dump(g, f)
        
    print("Done graph construction!")

if __name__ == "__main__":
    main()
