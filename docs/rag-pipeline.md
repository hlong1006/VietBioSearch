# Chạy pipeline VietBioSearch

Pipeline lấy câu truy vấn tiếng Việt, Anh và Trung, tìm tài liệu bằng BM25 và BGE-M3, kết hợp thứ hạng bằng RRF, rồi chọn các đoạn liên quan để xuất submission. Các tham số xếp hạng và định dạng đầu ra được chuyển từ notebook PA5 v2.

## Cấu trúc

~~~text
VietBioSearch/
├── src/
│   ├── __main__.py           # Điểm chạy python -m src
│   ├── cli.py                # Tham số dòng lệnh và đường dẫn
│   ├── pipeline.py           # Điều phối các bước
│   ├── config.py             # Tham số retrieval, embedding và chunking
│   ├── data.py               # Đọc Parquet và truy vấn JSON/CSV
│   ├── text.py               # Tách token và chia tài liệu
│   ├── bm25_index.py         # Tạo và nạp chỉ mục BM25
│   ├── embeddings.py         # Tạo và nạp embedding BGE-M3 theo shard
│   ├── retrieval.py          # Tìm và xếp hạng tài liệu theo ngôn ngữ
│   ├── scoring.py            # Chuẩn hóa điểm, RRF và số lượng kết quả
│   ├── chunk_selection.py    # Chấm điểm và chọn đoạn
│   ├── chunk_windows.py      # Cắt đoạn theo biên câu và giới hạn ký tự
│   ├── stages.py             # Chạy từng truy vấn và ghi checkpoint
│   ├── checkpoint.py         # Đọc, ghi và khôi phục trạng thái
│   └── export.py             # Xuất submission và gói checkpoint
├── docs/rag-pipeline.md
├── requirements.txt
├── Output/                   # Corpus đã chia shard và truy vấn mở rộng
├── Data_crawl/               # Dữ liệu thu thập
└── pa5-kaggle.ipynb           # Notebook chạy và fine-tune trên Kaggle
~~~

## Chạy tại máy

Cần Python 3.11 trở lên, GPU NVIDIA với PyTorch hỗ trợ CUDA và quyền truy cập trọng số `BAAI/bge-m3` trong cache hoặc qua Hugging Face. Chạy từ thư mục gốc của repo.

~~~sh
python -m pip install -r requirements.txt
python -m src --help
python -m src
~~~

Mặc định pipeline đọc `Output/corpus_shards/*.parquet`, ưu tiên `Output/queries_expanded.json` và dùng `Data_crawl/query.csv` hoặc `query.csv` dự phòng. Các shard cần ba cột `id`, `content` và `language`. CSV cung cấp truy vấn tiếng Việt. JSON có thể chứa thêm `vi_exp`, `en` và `zh`. Các thư mục dữ liệu được cung cấp riêng với mã nguồn.

~~~sh
python -m src --corpus-dir Output/corpus_shards --queries-file Output/queries_expanded.json --work-dir Output/pa5_output
~~~

Để chạy thử 5 truy vấn, dùng lệnh dưới đây. Lần đầu vẫn phải tải corpus và chuẩn bị chỉ mục cùng embedding.

~~~sh
python -m src --limit-queries 5 --skip-state-export
~~~

## Chạy trên Kaggle

Bật GPU và mở [pa5-kaggle.ipynb](../pa5-kaggle.ipynb). Gắn mã nguồn chứa `src/` và `requirements.txt`, cùng dataset chứa `corpus_shards/` và `queries_expanded.json` hoặc `query.csv`. Notebook có chế độ chạy thử trên một corpus nhỏ và bước fine-tune BGE-M3 nếu có file JSONL gắn nhãn `query`, `pos`, `neg`.

Pipeline tìm dữ liệu và checkpoint dưới `/kaggle/input`. Nếu có nhiều bộ dữ liệu, dùng `--input-root` hoặc chọn đường dẫn cụ thể bằng `--corpus-dir`, `--queries-file` và `--state-root`.

## Kết quả và tiếp tục phiên chạy

- Máy local ghi vào `Output/pa5_output/`. Chế độ giới hạn truy vấn dùng `Output/pa5_output_limit_N/` mặc định.
- Notebook ghi ZIP vào `/kaggle/working/pa5/` cho lần chạy đầy đủ và `/kaggle/working/pa5_smoke/` cho lần chạy thử. Lệnh CLI trực tiếp trên Kaggle dùng `/kaggle/working/pa5_output/` cho file trung gian và `/kaggle/working/` cho ZIP theo mặc định.
- `submission.zip` chứa `submission.json` với các trường `id`, `relevant_docs` và `relevant_chunks`. Dùng `--artifact-dir` để chọn nơi lưu các file ZIP.
- Để tiếp tục trên cùng máy, chạy lại với cùng thư mục làm việc và cùng bộ dữ liệu. Khi chuyển phiên Kaggle, đính kèm `pa5_state.zip` của phiên trước cùng bộ corpus và truy vấn. Có thể trỏ `--state-root` đến thư mục chứa ZIP hoặc checkpoint đã giải nén.
- Sửa tham số trong `src/config.py`. Dùng `--model` để chọn thư mục trọng số BGE-M3 sau fine-tune. Checkpoint được kiểm tra theo nội dung đầu vào và thiết lập embedding. Khi đổi dữ liệu hoặc model, hoặc khi đang dùng checkpoint cũ chưa có chữ ký đầu vào, chọn thư mục làm việc mới.
