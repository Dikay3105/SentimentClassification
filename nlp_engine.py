import streamlit as st

try:
    from underthesea import text_normalize
except ImportError:
    text_normalize = lambda x: x  # Nếu không có underthesea

from transformers import pipeline

MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"

# Từ điển viết tắt / không dấu
REPLACEMENTS = {
    # 1 từ
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hk": "không",
    "kp": "không phải",
    "dc": "được",
    "đc": "được",
    "j": "gì",
    "cj": "cái gì",
    "thik": "thích",
    "ntn": "như thế nào",
    "nma": "nhưng mà",
    "vs": "với",
    "vl": "vãi",
    "vkl": "vãi",
    "vch": "vãi chưởng",
    "cx": "cũng",
    "mk": "mình",
    "mik": "mình",
    "bn": "bạn",
    "bh": "bao giờ",
    "hnay": "hôm nay",
    "hqua": "hôm qua",
    "dz": "dễ",
    "z": "v",
    "zay": "vậy",
    "biet": "biết",
    "kieu": "kiểu",
    "chon": "chọn",
    "oki": "ok",
    "oke": "ok",
    "cute": "dễ thương",
    "xau": "xấu",
    "dep": "đẹp",
    "huhu": "buồn",
    "hik": "buồn",
    "haiz": "chán",
    "hehe": "vui",
    # cụm từ
    "khá ok": "khá ổn",
    "hom nay": "hôm nay",
    "hom qua": "hôm qua",
    "vs lại": "với lại",
    "chán v": "chán vậy",
    "đỉnh v": "đỉnh vậy",
    "đuối v": "đuối vậy",
    "quá đã": "rất thích",
    "de thuong": "dễ thương",
    "xịn xò": "tốt",
}


@st.cache_resource
def load_classifier():
    """Tải và cache mô hình Transformer."""
    return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)
    # return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)


classifier = load_classifier()


def replace_words(text, replacements):
    """Thay thế các từ viết tắt / không dấu theo từ điển."""
    words = text.split()
    new_words = [replacements.get(word.lower(), word) for word in words]
    return " ".join(new_words)


def classify_sentiment(text):
    """Phân loại cảm xúc tiếng Việt với từ điển không dấu / viết tắt."""
    if len(text.strip()) < 5:
        return "ERROR", "Câu quá ngắn, vui lòng nhập ít nhất 5 ký tự."

    # Thay thế từ viết tắt / không dấu
    text = replace_words(text, REPLACEMENTS)

    # Chuẩn hóa văn bản
    processed_text = text_normalize(text)

    result = classifier(processed_text)[0]
    label = result["label"]
    score = result["score"]

    # Ánh xạ nhãn cho mô hình wonrax/phobert-base-vietnamese-sentiment
    if label in ["NEG", "LABEL_0"]:
        sentiment = "NEGATIVE (Tiêu cực)"
    elif label in ["POS", "LABEL_1"]:
        sentiment = "POSITIVE (Tích cực)"
    else:
        sentiment = "NEUTRAL (Trung tính)"

    # Kiểm tra xác suất để gán nhãn NEUTRAL nếu độ tin cậy thấp
    if score < 0.6 and sentiment != "NEUTRAL (Trung tính)":
        sentiment = "NEUTRAL (Trung tính)"

    display_text = f"{sentiment} (Độ tin cậy: {score:.2f})"
    return sentiment, display_text


def set_style(sentiment):
    if "POSITIVE" in sentiment:
        return ":green[TÍCH CỰC]"
    elif "NEGATIVE" in sentiment:
        return ":red[TIÊU CỰC]"
    else:
        return ":orange[TRUNG TÍNH]"
