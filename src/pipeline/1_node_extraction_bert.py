import os
import pandas as pd
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

import re

def clean_text(text):
    text = str(text)
    # Xóa URL
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Xóa ký tự đặc biệt vô nghĩa
    text = re.sub(r'[^\w\s#@]', '', text)
    return text.strip()

def main():
    data_dir = "../data/COVID19"
    out_dir = os.path.join(data_dir, "completed_data")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading COVID19 datasets...")
    train_df = pd.read_csv(os.path.join(data_dir, "Constraint_Train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "Constraint_Val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "english_test_with_labels.csv"))
    
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Map nhãn dán: fake = 1, real = 0
    df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() == 'fake' else 0)
    
    # Cột chứa văn bản của bộ này tên là 'tweet'
    df['text'] = df['tweet']
    
    # Tiền xử lý (Cleaning & Filtering)
    print("Cleaning and filtering data...")
    df['text'] = df['text'].apply(clean_text)
    
    # Lọc bỏ tin rác < 5 từ
    initial_len = len(df)
    df = df[df['text'].apply(lambda x: len(x.split()) >= 5)].reset_index(drop=True)
    print(f"Dropped {initial_len - len(df)} short/spam tweets. Total valid tweets: {len(df)}")
    
    # XUẤT FILE MINH CHỨNG CHO HỘI ĐỒNG
    csv_out_path = os.path.join(data_dir, "AAAI2021_COVID19_Cleaned_Merged.csv")
    df[['text', 'label']].to_csv(csv_out_path, index=False, encoding='utf-8')
    print(f"Đã xuất file minh chứng gộp ra: {csv_out_path}")
    
    # Lấy sample nhỏ để chạy nhanh (bỏ comment để chạy full)
    # df = df.sample(200).reset_index(drop=True)
    
    texts = df['text'].astype(str).tolist()
    news_ids = [str(i) for i in range(len(texts))]  # Dùng index làm id cho dễ map
    
    # 2. Extract Entities using SpaCy
    print("Extracting entities...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("Model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        return
        
    news_entities = [] # list of lists of entities for each news
    all_entities = set()
    
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(texts)} articles for entities")
        # Giới hạn text length để tránh spacy chạy quá lâu
        doc = nlp(text[:5000]) 
        ents = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT']]
        ents = list(set(ents))
        news_entities.append(ents)
        all_entities.update(ents)
        
    entities = list(all_entities)
    print(f"Total distinct entities: {len(entities)}")
    
    # 3. Topic Extraction using LDA
    print("Extracting topics with LDA...")
    tokenized_texts = [[word.lower() for word in doc.split()] for doc in texts]
    dictionary = Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    num_topics = 50
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=2, random_state=42)
    
    # 4. Generate Features (Nâng cấp LLM/BERT)
    print("Generating Semantic Features with Sentence-Transformers...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Model 'sentence_transformers' not found. Please run: pip install sentence-transformers")
        return
        
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # News features (BERT Embeddings 384-dim)
    print("  - Encoding News Texts...")
    news_features = sbert_model.encode(texts, show_progress_bar=True)
    
    # Entity Features (BERT Embeddings 384-dim)
    print("  - Encoding Entity Names...")
    entity_features = sbert_model.encode(entities, show_progress_bar=True)
    
    # Topic Features (BERT Embeddings 384-dim based on Top words)
    print("  - Encoding Topics...")
    topic_descriptions = []
    for i in range(num_topics):
        # Lấy 10 từ khóa hàng đầu của từng topic để tạo thành câu mô tả ngữ nghĩa
        top_words = lda_model.show_topic(i, topn=10)
        topic_words_str = " ".join([word for word, prob in top_words])
        topic_descriptions.append(f"topic about {topic_words_str}")
        
    topic_features = sbert_model.encode(topic_descriptions, show_progress_bar=True)
    
    # 5. Save Outputs
    print("Saving node features...")
    dataset_name = "AAAI2021_COVID19_fake_news"
    
    mapindex = {}
    
    # Save News
    with open(os.path.join(out_dir, f"{dataset_name}.content.text"), 'w', encoding='utf-8') as f:
        for i, news_id in enumerate(news_ids):
            feat_str = '\t'.join([str(x) for x in news_features[i]])
            f.write(f"{news_id}\t{feat_str}\n")
            mapindex[news_id] = int(news_id)
            
    # Save Entities
    current_id = len(news_ids)
    entity_id_map = {}
    with open(os.path.join(out_dir, f"{dataset_name}.content.entity"), 'w', encoding='utf-8') as f:
        for i, ent in enumerate(entities):
            ent_id_str = str(current_id + i)
            entity_id_map[ent] = ent_id_str
            feat_str = '\t'.join([str(x) for x in entity_features[i]])
            f.write(f"{ent_id_str}\t{feat_str}\n")
            mapindex[ent] = int(ent_id_str)
            
    # Save Topics
    current_id += len(entities)
    topic_id_map = {}
    with open(os.path.join(out_dir, f"{dataset_name}.content.topic"), 'w', encoding='utf-8') as f:
        for i in range(num_topics):
            topic_str = f"topic_{i}"
            topic_id_str = str(current_id + i)
            topic_id_map[topic_str] = topic_id_str
            feat_str = '\t'.join([str(x) for x in topic_features[i]])
            f.write(f"{topic_id_str}\t{feat_str}\n")
            mapindex[topic_str] = int(topic_id_str)
            
    # Save mapindex.txt
    with open(os.path.join(out_dir, "mapindex.txt"), 'w', encoding='utf-8') as f:
        for old_idx, new_idx in mapindex.items():
            f.write(f"{old_idx}\t{new_idx}\n")
            
    # Save intermediate data for Graph Construction
    import pickle
    with open(os.path.join(out_dir, "intermediate_data.pkl"), 'wb') as f:
        pickle.dump({
            'news_ids': news_ids,
            'news_entities': news_entities,
            'entities': entities,
            'entity_id_map': entity_id_map,
            'topic_id_map': topic_id_map,
            'corpus': corpus,
            'lda_model': lda_model,
            'labels': df['label'].tolist()
        }, f)
        
    # Thống kê cho bài báo
    num_fake = df['label'].sum()
    print(f"Dataset COVID-19: #News {len(news_ids)} | #Entities {len(entities)} | #Topics {num_topics} | #Fake {num_fake}")
    print("Done node extraction!")

if __name__ == "__main__":
    main()
