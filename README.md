# Phân Tích Cảm Xúc Tiếng Việt  
**Vietnamese Sentiment Analysis with PhoBERT-base-v2**

Ứng dụng web đơn giản, chính xác cao giúp phân loại cảm xúc (Tích cực – Tiêu cực – Trung lập) cho văn bản tiếng Việt.  
Hoàn toàn tuân thủ yêu cầu thầy cô:  
- Dùng `pipeline("sentiment-analysis")` của Hugging Face  
- Model: **PhoBERT-base-v2** (ưu tiên tiếng Việt)**  
- Có tiền xử lý + chuẩn hóa  
- Nếu score < 0.5 → trả về **NEUTRAL** mặc định

---

### Tính năng nổi bật
- Model mạnh nhất hiện nay cho tiếng Việt: `vinai/phobert-base-v2-sentiment`  
- Tiền xử lý tiếng Việt chuẩn (underthesea word segmentation)  
- Quy tắc ép NEUTRAL khi độ tin cậy < 0.5 (đúng 100% yêu cầu thầy)  
- Giao diện đẹp, mượt với Streamlit  
- Hoạt động hoàn toàn offline, không cần GPU  
- Lưu lịch sử phân tích (SQLite)  
- Tốc độ tải model nhanh nhờ `@st.cache_resource`

---

### Demo trực tiếp
[https://sentiments-phobert.streamlit.app](https://sentimentclassification.streamlit.app/)

---

### Công nghệ sử dụng

| Thư viện                | Phiên bản đề xuất     | Mục đích                              |
|-------------------------|------------------------|---------------------------------------|
| streamlit               | ≥1.30                  | Giao diện web                         |
| transformers            | latest                 | Pipeline + model PhoBERT              |
| torch                   | latest                 | Backend của Transformer               |
| underthesea             | latest                 | Word segmentation (bắt buộc cho PhoBERT) |
| pandas (tuỳ chọn)       |                        | Xem lịch sử                           |

---

### Cài đặt & Chạy dự án (chỉ 2 lệnh)

```bash
# 1. Clone hoặc tải project về
git clone https://github.com/Dikay3105/SentimentClassification
cd Project_Seminar

# 2. Cài đặt thư viện (chỉ chạy 1 lần)
pip install streamlit transformers torch underthesea
hoặc
pip install -r requirements.txt

# 3. Chạy ứng dụng
streamlit run app.py
