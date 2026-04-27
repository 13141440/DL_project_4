# IconQA 项目笔记

## 数据可视化

项目提供了三种方式查看 IconQA 数据样本：

### 方式一：Jupyter Notebook（推荐）

在 VS Code 中打开 `view_examples.ipynb`，选择 `.venv` 作为 kernel，逐个 cell 运行即可看到带图片的样本预览。

### 方式二：静态 HTML

直接用浏览器打开 `view_examples.html`。如果在远程服务器上，可以启动 HTTP 服务：

```bash
.venv/bin/python -m http.server 8080
```

然后浏览器访问 `http://<服务器IP>:8080/view_examples.html`。

### 方式三：Streamlit 交互式浏览

```bash
.venv/bin/streamlit run view_data.py --server.port 8501
```

浏览器访问 `http://<服务器IP>:8501`，可以通过侧边栏切换训练集/验证集，输入样本编号逐条浏览。
