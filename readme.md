# HƯỚNG DẪN CHẠY THỰC NGHIỆM
**Đề tài:** Phát hiện Tin giả dựa trên Đồ thị Tri thức (Knowledge Graph) kết hợp Mô hình Ngôn ngữ lớn (LLM - BERT).
**Bộ dữ liệu (Dataset):** COVID-19 (AAAI-2021).

---

## BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG (Trên Server)

Đầu tiên, bạn cần cài đặt các thư viện cần thiết. Khuyến nghị sử dụng **Python 3.8+**.

1. Cài đặt các thư viện Python:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric pandas numpy spacy gensim scikit-learn networkx requests tqdm sentence-transformers
```

2. Tải mô hình Ngôn ngữ của SpaCy (Dùng để trích xuất Thực thể):
```bash
python -m spacy download en_core_web_sm
```

---

## BƯỚC 2: CHUẨN BỊ DỮ LIỆU TỪ TỪ ĐIỂN WIKIDATA5M

Bạn cần tải các Vector Tri thức (Knowledge Graph Embeddings) khổng lồ đã được Pre-train sẵn từ dự án OpenKE/DGL.

1. Hãy đảm bảo bạn có thư mục `data/pretrained/` trong source code.
2. Tải 2 file sau từ mạng về và bỏ vào thư mục `data/pretrained/`:
   - `wikidata5m_transe.txt` (Dung lượng > 4GB, chứa Vector 512-chiều)
   - *Lưu ý: Bạn cũng cần đảm bảo đã có sẵn thư mục `data/COVID19/` chứa 3 file CSV ban đầu (Constraint_Train, Constraint_Val, english_test).*
   - link tải: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/README.md

---

## BƯỚC 3: CHẠY QUY TRÌNH (PIPELINE)

Quy trình đã được lập trình tự động hóa hoàn toàn. Bạn chỉ cần chạy đúng 4 lệnh sau theo thứ tự từ trên xuống dưới:

### Lệnh 1: Tiền xử lý & Trích xuất Đặc trưng (Node Extraction)
- **Chức năng:** Đọc 3 file CSV, dọn rác văn bản (xóa URL), loại bỏ các tin siêu ngắn (< 5 từ). Dùng SpaCy tìm Thực thể, dùng LDA tìm Chủ đề, và dùng BERT (SentenceTransformers) để tạo ma trận Vector ngữ nghĩa. Cuối cùng, xuất ra một file minh chứng `AAAI2021_COVID19_Cleaned_Merged.csv`.
```bash
python src/pipeline/1_node_extraction_bert.py
```
*(Kết thúc lệnh này, màn hình sẽ in ra một bảng thống kê 1 dòng cực kỳ rõ ràng để bạn đưa vào báo cáo).*

### Lệnh 2: Xây dựng Cấu trúc Đồ thị (Graph Construction)
- **Chức năng:** Đọc các Thực thể và Chủ đề từ Lệnh 1, xây dựng một siêu Mạng nhện chằng chịt các đường nối (Edges) giữa Bài báo - Thực thể và Bài báo - Chủ đề.
```bash
python src/pipeline/2_graph_construction.py
```

### Lệnh 3: Nhúng Đồ thị Tri thức (Knowledge Graph Embedding)
- **Chức năng:** Lấy các Thực thể đi tra từ điển Wikidata (lấy ID). Sau đó mở file Vector 4GB ra để lấy Vector 512-chiều cấu trúc. Kết hợp thêm Vector 128-chiều ngữ nghĩa từ BERT.
```bash
python src/pipeline/3_kg_embedding.py
```

### Lệnh 4: Huấn luyện Mô hình GNN (Train & Evaluate)
- **Chức năng:** Ném toàn bộ Đồ thị, Vector Đặc trưng vào Mạng Nơ-ron Đồ thị (GNN). Áp dụng cơ chế Masking (Transductive Learning) chia 80% Train và 20% Test để chấm điểm.
```bash
python src/main.py --dataset AAAI
```

---
