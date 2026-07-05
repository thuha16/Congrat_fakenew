import os
import pickle
import pandas as pd
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from tqdm import tqdm

def main():
    data_dir = "../data/FakeNewsNet"
    
    # 1. Load Data
    print("Loading datasets...")
    pf_fake = pd.read_csv(os.path.join(data_dir, "PolitiFact_fake_news_content.csv"))
    pf_real = pd.read_csv(os.path.join(data_dir, "PolitiFact_real_news_content.csv"))
    bf_fake = pd.read_csv(os.path.join(data_dir, "BuzzFeed_fake_news_content.csv"))
    bf_real = pd.read_csv(os.path.join(data_dir, "BuzzFeed_real_news_content.csv"))
    
    pf_fake['label'] = 1
    pf_real['label'] = 0
    bf_fake['label'] = 1
    bf_real['label'] = 0
    
    df = pd.concat([pf_fake, pf_real, bf_fake, bf_real], ignore_index=True)
    
    # Drop rows without text
    df = df.dropna(subset=['text']).reset_index(drop=True)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # 2. Extract Entities using Spacy
    print("Extracting entities...")
    nlp = spacy.load("en_core_web_sm")
    
    entities = set()
    news_entities = []
    
    for text in tqdm(texts, desc="NER Processing"):
        doc = nlp(text)
        ents = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT']]
        entities.update(ents)
        news_entities.append(ents)
        
    entities = list(entities)
    
    # 3. Extract Topics using LDA
    print("Extracting topics (LDA)...")
    tokenized_texts = [preprocess_string(t) for t in texts]
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    num_topics = 50
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=2, random_state=42)
    
    # 4. Generate Features (BASELINE: DOC2VEC)
    print("Generating Features with Doc2vec...")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_texts)]
    d2v_model = Doc2Vec(documents, vector_size=128, window=5, min_count=2, workers=4, epochs=10)
    
    news_features = []
    for i in range(len(texts)):
        vec = d2v_model.dv[i]
        news_features.append(vec)
    news_features = np.array(news_features)
    
    # Entity Features (Random init 64-dim)
    np.random.seed(42)
    entity_features = np.random.normal(0, 0.1, (len(entities), 64))
    
    # Topic Features (One-hot 50-dim)
    topic_features = np.eye(num_topics)
    
    # 5. Save Outputs
    print("Saving node features...")
    out_dir = os.path.join(data_dir, "completed_data")
    os.makedirs(out_dir, exist_ok=True)
    
    np.savetxt(os.path.join(out_dir, "FakeNewsNet.content.text"), news_features, fmt='%.6f', delimiter='\t')
    np.savetxt(os.path.join(out_dir, "FakeNewsNet.content.entity"), entity_features, fmt='%.6f', delimiter='\t')
    np.savetxt(os.path.join(out_dir, "FakeNewsNet.content.topic"), topic_features, fmt='%.6f', delimiter='\t')
    
    # Save intermediate data for Graph Construction
    with open(os.path.join(out_dir, "intermediate_data.pkl"), 'wb') as f:
        pickle.dump({
            'texts': texts,
            'labels': labels,
            'entities': entities,
            'news_entities': news_entities,
            'corpus': corpus,
            'lda_model': lda_model,
            'num_topics': num_topics
        }, f)
        
    print("Done! Baseline data extraction completed.")

if __name__ == "__main__":
    main()
