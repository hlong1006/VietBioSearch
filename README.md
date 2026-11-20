# VietBioSearch

**VietBioSearch** là dự án cá nhân về truy xuất tài liệu và đoạn văn y sinh đa ngôn ngữ, hỗ trợ truy vấn tiếng Việt, tiếng Anh và tiếng Trung. Hệ thống kết hợp tìm kiếm từ khóa BM25 với embedding BGE-M3, xếp hạng lại tài liệu, hợp nhất thứ hạng bằng RRF rồi chọn các đoạn liên quan.

Dự án phát triển từ pipeline ViBioMIR PA5 và xuất kết quả dưới dạng `submission.zip` tương thích với PA5.

[Dữ liệu crawl trên Google Drive](https://drive.google.com/drive/u/0/folders/1QsbxgllgQ9zMk68ucujwpuEm8JYXGY8d) · [Notebook Kaggle](pa5-kaggle.ipynb) · [Tài liệu kỹ thuật](docs/rag-pipeline.md)

## Chuẩn bị dữ liệu

Tải dữ liệu crawl từ Google Drive và lưu vào `Data_crawl/` nếu làm việc trên máy. Điểm bắt đầu của pipeline là **corpus Parquet đã xử lý** và **tệp truy vấn**. Nếu thư mục Drive chỉ có dữ liệu crawl, cần chuyển corpus sang Parquet trước khi chạy. Corpus Parquet và truy vấn mở rộng được chuẩn bị bên ngoài repo.

| Đầu vào | Vị trí mặc định khi chạy local | Định dạng |
| --- | --- | --- |
| Corpus đã xử lý | `Output/corpus_shards/*.parquet` | Các cột `id`, `content`, `language`. Mỗi tài liệu có ID duy nhất và mã ngôn ngữ `vi`, `en` hoặc `zh`. |
| Truy vấn mở rộng | `Output/queries_expanded.json` | JSON dạng đối tượng theo ID hoặc danh sách các đối tượng có trường `id`. Nội dung có thể gồm `vi_orig`, `vi_exp`, `en`, `zh`. |
| Truy vấn tiếng Việt | `Data_crawl/query.csv` hoặc `query.csv` ở thư mục gốc | CSV có cột `id,query`. Khi dùng riêng CSV, pipeline truy xuất trong nhóm tài liệu tiếng Việt. |

ID truy vấn cần là số nguyên hoặc chuỗi biểu diễn số nguyên và phải duy nhất. Repo dùng `.gitignore` để bỏ qua dữ liệu crawl, output, trọng số và cache. [Quy tắc ignore](.gitignore)

## Chạy trên Kaggle

1. Import [`pa5-kaggle.ipynb`](pa5-kaggle.ipynb) vào Kaggle. Chọn **GPU** trong **Session options** và bật **Internet** nếu cần tải thư viện hoặc trọng số. [Hướng dẫn Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)
2. Dùng **Add Input** để gắn dữ liệu đã xử lý. Thư mục đầu vào cần có `corpus_shards/` và `queries_expanded.json` hoặc `query.csv`. Nếu có nhiều bộ dữ liệu, đặt `DATA_ROOT` hoặc chỉ định đường dẫn các tệp trong ô cấu hình.
3. Gắn mã nguồn chứa `src/` và `requirements.txt` cùng cấp. Đặt `SOURCE_DIR` nếu cần chọn một thư mục cụ thể. Với repo Git truy cập được từ Kaggle, có thể đặt `REPO_URL` để notebook clone mã nguồn.
4. Giữ `SMOKE_TEST = True` cho lần đầu. Notebook mặc định lấy tối đa 30 tài liệu mỗi ngôn ngữ và 3 truy vấn. Đặt `SMOKE_TEST = False` để chạy toàn bộ và lấy `submission.zip` trong `/kaggle/working/pa5/`.

Notebook mặc định dùng trọng số BGE-M3 có sẵn. Để fine-tune dense encoder, đặt `TRAIN_JSONL` tới tệp dữ liệu gắn nhãn. Mỗi dòng cần có `query` là chuỗi, còn `pos` và `neg` là danh sách đoạn văn liên quan và không liên quan. Nếu có nhãn đánh giá riêng, đặt `QRELS_JSON` tới JSON ánh xạ ID truy vấn sang danh sách ID tài liệu liên quan để tính Recall@10 và MRR@10 trong lần chạy đầy đủ. [Model BGE-M3](https://huggingface.co/BAAI/bge-m3) · [Định dạng dữ liệu fine-tune](https://bge-model.com/tutorial/7_Finetuning/7.1.2.html)

## Chạy tại máy

Cần Python 3.11 trở lên, GPU NVIDIA, PyTorch hỗ trợ CUDA và trọng số `BAAI/bge-m3` tải từ Hugging Face hoặc có sẵn trong cache. Chạy các lệnh từ thư mục gốc của repo. [Cài PyTorch với CUDA](https://pytorch.org/get-started/locally/)

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python -m src --help
```

Khi kiểm tra CUDA trả về `True` và dữ liệu nằm đúng vị trí mặc định, chạy:

```bash
python -m src
```

Để chỉ định đường dẫn và chạy 5 truy vấn đầu tiên, dùng:

```bash
python -m src --corpus-dir Output/corpus_shards --queries-file Output/queries_expanded.json --limit-queries 5 --skip-state-export
```

Thay `--queries-file Output/queries_expanded.json` bằng `--query-csv Data_crawl/query.csv` nếu dùng CSV. `--limit-queries` chỉ giới hạn số truy vấn. Lần đầu CLI vẫn tạo chỉ mục và embedding cho toàn bộ corpus được truyền vào.

## Kết quả và checkpoint

| Tệp | Nội dung |
| --- | --- |
| `submission.zip` | Chứa `submission.json` với các trường `id`, `relevant_docs` và `relevant_chunks`. |
| `pa5_state.zip` | Chứa checkpoint, kết quả trung gian, chỉ mục BM25 và embedding có sẵn để tiếp tục xử lý. |

- **Local:** mặc định ghi vào `Output/pa5_output/`. Khi dùng `--limit-queries 5`, thư mục mặc định là `Output/pa5_output_limit_5/`.
- **Notebook Kaggle:** lần chạy đầy đủ ghi ZIP vào `/kaggle/working/pa5/`. Lần chạy thử ghi `submission.zip` vào `/kaggle/working/pa5_smoke/`.
- **Tiếp tục:** trên máy, chạy lại với cùng `--work-dir`, corpus, truy vấn và model. Trên Kaggle, gắn `pa5_state.zip` từ phiên trước và đặt `STATE_ROOT` tới thư mục chứa ZIP. Khi thay corpus, truy vấn hoặc model, dùng thư mục làm việc mới. Trong notebook, đổi `RUN_NAME` để tạo thư mục mới.

Tham số truy xuất, embedding và chọn đoạn nằm trong [`src/config.py`](src/config.py). Các tùy chọn đường dẫn và cách khôi phục được mô tả thêm trong [tài liệu pipeline](docs/rag-pipeline.md).
