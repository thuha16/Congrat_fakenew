# Hướng dẫn chạy Pipeline cho bộ dữ liệu LIAR

Tài liệu này chứa các câu lệnh tuần tự để chạy mô hình phát hiện tin giả Congrat trên bộ dữ liệu **Liar**. Hãy mở Terminal, đảm bảo bạn đang đứng ở thư mục gốc của dự án (`Congrat-main`) và chạy lần lượt các lệnh dưới đây.

## Tiền yêu cầu
Đảm bảo bạn đã tải 2 file của Wikidata (Khối có chữ "Wikidata" với file embeddings >4GB và file entity ids 360MB) và đặt chúng vào đúng thư mục `data/pretrained/`.

---

### Bước 1: Tiền xử lý, Dọn rác và Trích xuất Đặc trưng
Lệnh này sẽ tự động đọc file `.tsv` của Liar, làm sạch văn bản, chuyển 6 nhãn về dạng Nhị phân (Fake/Real), và sử dụng mô hình `MPNet` để chuyển đổi câu văn thành các Vector 768-chiều.

```bash
python src/pipeline/1_node_extraction_bert.py --dataset Liar
```
*Đầu ra: 8 file chứa đặc trưng văn bản, thực thể, chủ đề... được lưu gọn gàng tại thư mục `data/Liar/completed_data/`.*

### Bước 2: Xây dựng Đồ thị Mạng nhện
Lệnh này sẽ sử dụng kết quả của Bước 1 để tính toán sự tương đồng cosine và tạo ra các cạnh (edges) nối các thực thể, bài báo và chủ đề lại với nhau tạo thành một Đồ thị.

```bash
python src/pipeline/2_graph_construction.py --dataset Liar
```
*Đầu ra: Sinh ra file `model_network_handled.pkl` chứa cấu trúc Đồ thị.*

### Bước 3: Nhúng Đồ thị Tri thức (Knowledge Graph Embedding)
Lệnh này sẽ quét mã Q-node của các thực thể, tra cứu chúng trong kho Wikidata (file >4GB bạn đã tải) và kết hợp với ngữ nghĩa từ MPNet.

```bash
python src/pipeline/3_kg_embedding.py --dataset Liar
```
*Đầu ra: Sinh ra file `ent_attr_kg_transe.pkl` chứa các Vector tri thức uyên thâm.*

### Bước 4: Huấn luyện và Đánh giá (Huấn luyện Mạng Đồ thị - GATv2)
Lệnh cuối cùng này sẽ nạp tất cả dữ liệu ở trên vào mô hình GNN sâu 2 lớp (với cơ chế Dropout chống học vẹt) để bắt đầu huấn luyện. Nó sẽ lặp lại 10 lần (10 runs) để lấy kết quả khách quan nhất.

```bash
python src/main.py --dataset Liar
```
*Đầu ra: Bảng tổng kết kết quả điểm số Mean, Std, Min, Max cực kỳ chuyên nghiệp sẽ được in ra màn hình Terminal và lưu vào file `Para_analysis.txt`.*
