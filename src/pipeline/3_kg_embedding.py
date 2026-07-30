import os
import argparse
import pickle
import numpy as np
import requests
import urllib.parse
from tqdm import tqdm

WIKIDATA_USER_AGENT = "Congrat-fakenews-thesis/1.0 (https://github.com/thuha16/Congrat_fakenew)"

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
    # Wikidata's API now rejects requests with no User-Agent (HTTP 403, see
    # https://phabricator.wikimedia.org/T400119), which silently made every
    # lookup fail and fall back to random vectors.
    headers = {"User-Agent": WIKIDATA_USER_AGENT}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
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

def load_wikidata_embeddings(entity2id_path, entity2vec_path, target_qnodes, dim):
    """
    Tra cứu vector TransE thực từ bộ Wikidata đầy đủ do OpenKE pre-train:
    entity2id.txt ánh xạ Q-node -> chỉ số dòng, entity2vec.bin là mảng nhị phân
    float32 (memmap để không load toàn bộ >8GB vào RAM).
    """
    embeddings = {}
    if not os.path.exists(entity2id_path) or not os.path.exists(entity2vec_path):
        print(f"Warning: Không tìm thấy {entity2id_path} hoặc {entity2vec_path}. Sẽ dùng vector ngẫu nhiên.")
        return embeddings

    print(f"Đang đọc {entity2id_path} để tra cứu {len(target_qnodes)} Q-node...")
    qnode_to_idx = {}
    with open(entity2id_path, 'r', encoding='utf-8') as f:
        num_entities = int(f.readline().strip())
        for line in tqdm(f, total=num_entities):
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[0] in target_qnodes:
                qnode_to_idx[parts[0]] = int(parts[1])

    vec_mmap = np.memmap(entity2vec_path, dtype=np.float32, mode='r').reshape(-1, dim)
    for qnode, idx in qnode_to_idx.items():
        embeddings[qnode] = np.array(vec_mmap[idx])

    return embeddings

def main(args):
    data_dir = f"../data/{args.dataset}/completed_data"

    # ĐƯỜNG DẪN TỚI BỘ WIKIDATA ĐẦY ĐỦ DO OPENKE PRE-TRAIN (entity2id.txt + entity2vec.bin)
    WIKIDATA_ENTITY2ID_FILE = "../data/pretrained/Wikidata/knowledge graphs/entity2id.txt"
    WIKIDATA_ENTITY2VEC_FILE = "../data/pretrained/Wikidata/embeddings/dimension_100/transe/entity2vec.bin"
    WIKIDATA_DIM = 100

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

    print(f"2. Đang tra cứu Wikidata Embeddings ({WIKIDATA_DIM}-dim)...")
    wiki_vectors = load_wikidata_embeddings(WIKIDATA_ENTITY2ID_FILE, WIKIDATA_ENTITY2VEC_FILE, target_qnodes, WIKIDATA_DIM)
    
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
            kg_transe[ent] = np.random.normal(0, 0.1, WIKIDATA_DIM).astype(np.float32)
            
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
    parser = argparse.ArgumentParser(description="Pipeline 3: Knowledge Graph Embedding")
    parser.add_argument('--dataset', type=str, default='COVID19', help='Dataset to process (e.g., COVID19, Liar)')
    args = parser.parse_args()
    main(args)
