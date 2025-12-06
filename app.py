import streamlit as st
from nlp_engine import classify_sentiment, set_style
from db_engine import init_db, save_result, load_history


def main():
    init_db()

    st.title("Trợ Lý Phân Loại Cảm Xúc Tiếng Việt")
    st.markdown("Sử dụng **Transformer PhoBERT** và lưu trữ **SQLite**.")

    # --- Nhập liệu ---
    st.header("1. Nhập liệu")
    user_input = st.text_area(
        "Nhập câu/đoạn văn bản tiếng Việt (mỗi dòng là 1 câu):", ""
    )

    if st.button("Phân loại Cảm xúc"):
        lines = [line.strip() for line in user_input.splitlines() if line.strip()]
        if not lines:
            st.warning("Vui lòng nhập ít nhất 1 câu để phân loại.")
        else:
            st.subheader("2. Kết quả Phân loại")
            for i, line in enumerate(lines, start=1):
                with st.spinner(f"Đang phân tích câu {i}/{len(lines)}..."):
                    final_sentiment, display_text = classify_sentiment(line)
                    st.markdown(f"**Câu {i}:** *{line}*")
                    st.markdown(
                        f"**Nhãn Cảm xúc:** {set_style(final_sentiment)} (_{display_text}_)"
                    )

                    if final_sentiment != "ERROR":
                        save_result(line, final_sentiment)
                    else:
                        st.error(display_text)

            st.success("Đã phân loại xong tất cả câu và lưu vào lịch sử.")

    st.markdown("---")

    # --- Lịch sử ---
    st.header("3. Lịch sử phân loại (SQLite)")

    # Mặc định: chỉ lấy 50 dòng
    history_df = load_history(limit=50)

    # Nút xem thêm
    show_all = st.checkbox("📄 Xem toàn bộ lịch sử")

    if show_all:
        history_df = load_history(limit=None)  # tải tất cả

    if not history_df.empty:
        history_df = history_df.rename(
            columns={
                "text": "Văn bản",
                "sentiment": "Cảm xúc",
                "timestamp": "Thời gian",
            }
        ).drop(columns=["id"])
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("Chưa có lịch sử phân loại nào được lưu.")


if __name__ == "__main__":
    main()
