import streamlit as st
import datasets

st.set_page_config(page_title="IconQA 数据查看器", layout="wide")
st.title("IconQA 数据查看器")

split = st.sidebar.radio("选择数据集", ["训练集", "验证集"])
path = "data/icon-qa-train.arrow" if split == "训练集" else "data/icon-qa-val.arrow"

ds = datasets.Dataset.from_file(path)
st.sidebar.write(f"共 **{len(ds)}** 条样本")

idx = st.sidebar.number_input("样本编号", min_value=0, max_value=len(ds) - 1, value=0)
sample = ds[idx]

st.subheader(f"样本 #{idx}")

st.markdown(f"**问题:** {sample['question']}")
st.markdown(f"**选项:** {sample['choices']}")
if sample["answer"] is not None:
    st.markdown(f"**答案:** {sample['answer']}")

col1, col2, col3 = st.columns(3)
with col1:
    st.caption("题目图片 (query_image)")
    st.image(sample["query_image"], use_container_width=True)
with col2:
    st.caption("选项图片 0 (choice_image_0)")
    st.image(sample["choice_image_0"], use_container_width=True)
with col3:
    st.caption("选项图片 1 (choice_image_1)")
    st.image(sample["choice_image_1"], use_container_width=True)
