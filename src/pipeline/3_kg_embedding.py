import os
import pickle
import numpy as np
import requests
import urllib.parse
from tqdm import tqdm

def search_wikidata(entity_name):
    """
    Sử dụng Wikidata API để tìm ID (Q-node) của một thực thể.
    """
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if data.get('search') and len(data['search']) > 0:
            return data['search'][0]['id']  # Lấy kết quả phù hợp nhất
    except Exception as e:
        pass
    return None

def get_dbpedia_resource(entity_name):
    """
    DBpedia ID thường lấy trực tiếp từ Wikipedia page title.
    """
    title = urllib.parse.quote(entity_name.replace(' ', '_').title())
    return f"http://dbpedia.org/resource/{title}"

def load_pretrained_embeddings(filepath, entity_ids, dim):
    """
    Đọc file text chứa Pre-trained embeddings (từng dòng) để tra cứu vector.
    Tránh load toàn bộ file vào RAM để không bị tràn bộ nhớ.
    """
    embeddings = {}
    if not os.path.exists(filepath):
        print(f"Warning: Không tìm thấy file {filepath}. Sẽ dùng vector ngẫu nhiên.")
        return embeddings
        
    print(f"Đang quét file {filepath} để tìm {len(entity_ids)} entities...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().split()
            if len(parts) > dim:
                ent_id = parts[0]
                if ent_id in entity_ids:
                    # Lấy dim phần tử cuối cùng làm vector
                    vec = np.array([float(x) for x in parts[-dim:]], dtype=np.float32)
                    embeddings[ent_id] = vec
                    
                    # Tối ưu: Nếu đã tìm đủ vector thì dừng quét
                    if len(embeddings) == len(entity_ids):
                        break
    return embeddings

def main():
    data_dir = "../data/COVID19/completed_data"
    
    # ĐƯỜNG DẪN TỚI FILE WIKIDATA TRÊN SERVER (Tải từ OpenKE)
    WIKIDATA_PRETRAINED_FILE = "../data/pretrained/wikidata5m_transe.txt"
    
    print("Loading intermediate data...")
    with open(os.path.join(data_dir, "intermediate_data.pkl"), 'rb') as f:
        data = pickle.load(f)
        
    entities = data['entities']
    
    print("1. Đang thực hiện Entity Linking (Tìm mã ID Wikidata)...")
    wikidata_mapping = {}
    
    for ent in tqdm(entities, desc="Entity Linking"):
        q_node = search_wikidata(ent)
        if q_node:
            wikidata_mapping[ent] = q_node
            
    target_qnodes = set(wikidata_mapping.values())
    
    print("2. Đang tra cứu Wikidata Embeddings (512-dim)...")
    wiki_vectors = load_pretrained_embeddings(WIKIDATA_PRETRAINED_FILE, target_qnodes, 512)
    
    print("3. Khởi tạo Semantic Knowledge Branch (256-dim) bằng BERT MPNet...")
    try:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer('all-mpnet-base-v2')
        print("Đang mã hóa ý nghĩa tên thực thể...")
        semantic_embeddings = sbert_model.encode(entities, show_progress_bar=True)
    except ImportError:
        print("Lỗi: Chưa cài đặt sentence-transformers. Hãy chạy: pip install sentence-transformers")
        return
        
    print("4. Tổng hợp vector cuối cùng...")
    kg_transe = {}
    dbpedia_transe = {} # Đã được đổi tên thành Semantic Branch
    
    found_wiki = 0
    
    for i, ent in enumerate(entities):
        # Wikidata Branch (Cấu trúc)
        q_node = wikidata_mapping.get(ent)
        if q_node and q_node in wiki_vectors:
            kg_transe[ent] = wiki_vectors[q_node]
            found_wiki += 1
        else:
            kg_transe[ent] = np.random.normal(0, 0.1, 512).astype(np.float32)
            
        # Semantic Branch (Ngữ nghĩa LLM) - Lấy 256 chiều đầu tiên của MPNet (768-dim)
        dbpedia_transe[ent] = semantic_embeddings[i][:256].astype(np.float32)
            
    print(f"Wikidata: Tìm thấy {found_wiki}/{len(entities)} vector thực tế.")
    print(f"Semantic Branch: 100% sử dụng BERT Embeddings.")
    
    print("Lưu dữ liệu cho GNN model...")
    with open(os.path.join(data_dir, "ent_attr_kg_transe.pkl"), 'wb') as f:
        pickle.dump(kg_transe, f)
        
    with open(os.path.join(data_dir, "ent_attr_DBpedia_transe.pkl"), 'wb') as f:
        pickle.dump(dbpedia_transe, f)
        
    print("Đã hoàn thành Knowledge Graph Embedding (Real Wikidata + LLM Semantic)!")

if __name__ == "__main__":
    main()
