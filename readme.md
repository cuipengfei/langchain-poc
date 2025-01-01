# 代码解释

## 概述

这是一个基于LangChain框架的智能问答系统，采用RAG (检索增强生成) 技术实现。系统通过向量化存储文档内容，结合大语言模型实现智能问答。代码架构清晰，包含以下核心模块：

### 1. 基础
- `environment_loader.py`: 环境配置管理
- `logging_setup.py`: 日志系统配置
- `token_counter.py`: Token计数工具

### 2. 功能
- `vector_store_setup.py`: 向量存储实现
- `chat_model_setup.py`: 聊天模型配置
- `init-vector.py`: 文档处理与向量化
- `poc.py`: 主程序逻辑

## 核心概念实现

### RAG技术实现
系统通过以下步骤实现RAG：
1. 文档加载与分块：使用`TextLoader`和`RecursiveCharacterTextSplitter`
2. 向量化存储：使用`DashScopeEmbeddings`转换文本为向量
3. 相似度检索：利用`Chroma`向量数据库进行检索
4. 结合上下文生成回答：使用`ChatTongyi`模型

```python
# 向量存储示例
vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=DashScopeEmbeddings(model="text-embedding-v3"),
    persist_directory="vectordb"
)
```

## 相关概念解释

### RAG（Retrieval-augmented generation，检索增强生成）

RAG是一种结合检索和生成的技术，通过检索相关文档来增强生成模型的回答能力。代码中通过向量存储和相似性搜索实现了这一点。

### 向量

向量是AI算法处理的基本单位，通过将文本转换为向量，可以计算向量之间的相似度，实现语义匹配。代码中使用了`DashScopeEmbeddings`来生成向量。

### 向量相似度计算

许多AI算法处理的是向量，通过计算向量之间的相似度来实现语义匹配。代码中使用了向量存储和相似性搜索来实现这一点。

### Token

Token是文本的基本单位，通常是一个单词或子词。代码中使用了`tiktoken`库来计算token数量，确保聊天历史不会超过模型的上下文窗口大小。

### Embedding

Embedding是将文本转换为向量的过程，代码中使用了`DashScopeEmbeddings`来生成文本的向量表示。

## RAG与模型微调

RAG和模型微调都可以解决将核心业务数据放在提示词里还是放在模型里的问题。代码中通过RAG技术实现了这一点。