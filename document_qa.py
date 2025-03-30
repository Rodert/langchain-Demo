'''
Author: javapub iswangshiyu@foxmail.com
Date: 2025-03-30 19:00:00
LastEditors: javapub iswangshiyu@foxmail.com
LastEditTime: 2025-03-30 17:37:56
FilePath: /langchain-Demo/document_qa.py
Description: LangChain 文档问答示例
'''
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def load_and_process_documents(file_path):
    """加载和处理文档"""
    print(f"加载文档: {file_path}")
    
    # 根据文件类型选择加载器
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")
    
    # 加载文档
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档片段")
    
    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"分割成 {len(splits)} 个文本块")
    
    return splits

def create_vector_store(splits):
    """创建向量存储"""
    print("创建向量存储...")
    
    # 使用Ollama的嵌入模型
    embeddings = OllamaEmbeddings(model="llama3:latest")
    
    # 创建向量存储
    vector_store = FAISS.from_documents(splits, embeddings)
    print("向量存储创建完成")
    
    return vector_store

def create_qa_chain(vector_store):
    """创建问答链"""
    print("创建问答链...")
    
    # 使用Ollama的LLM
    llm = OllamaLLM(model="llama3:latest")
    
    # 创建自定义提示模板
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
    
上下文:
{context}

问题: {question}

请用中文回答:"""
    
    QA_PROMPT = PromptTemplate.from_template(template)
    
    # 创建检索QA链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    
    print("问答链创建完成")
    
    return qa_chain

def ask_questions(qa_chain):
    """提问并获取回答"""
    print("\n=== 文档问答系统 ===")
    print("输入'退出'或'exit'结束对话")
    print("="*50)
    
    while True:
        question = input("\n问题: ")
        
        if question.lower() in ['退出', 'exit', 'quit']:
            break
        
        # 获取回答
        result = qa_chain.invoke({"query": question})
        
        print("\n回答:")
        print(result["result"])
        
        # 显示来源文档
        print("\n来源文档:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"文档 {i+1}:")
            print(f"内容: {doc.page_content[:150]}...")
            print(f"元数据: {doc.metadata}")
            print()

def main():
    """主函数"""
    # 示例：创建一个简单的文本文件用于演示
    sample_file = "sample_document.txt"
    
    # 如果文件不存在，创建一个示例文件
    if not os.path.exists(sample_file):
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("""# 人工智能简介

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。

## 机器学习

机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需明确编程。机器学习算法可以分为三类：
1. 监督学习：使用标记的数据进行训练
2. 无监督学习：使用未标记的数据寻找模式
3. 强化学习：通过与环境交互学习最优策略

## 深度学习

深度学习是机器学习的一个子集，使用多层神经网络处理复杂任务。深度学习在以下领域取得了显著成果：
- 计算机视觉
- 自然语言处理
- 语音识别
- 游戏

## 大语言模型

大语言模型（LLM）是一种基于深度学习的模型，专门用于处理和生成自然语言。它们通过分析大量文本数据学习语言模式。

著名的大语言模型包括：
- GPT（由OpenAI开发）
- LLaMA（由Meta开发）
- Claude（由Anthropic开发）

这些模型能够执行各种任务，如文本生成、翻译、摘要和问答。
""")
        print(f"已创建示例文件: {sample_file}")
    
    # 加载和处理文档
    splits = load_and_process_documents(sample_file)
    
    # 创建向量存储
    vector_store = create_vector_store(splits)
    
    # 创建问答链
    qa_chain = create_qa_chain(vector_store)
    
    # 提问
    ask_questions(qa_chain)

if __name__ == "__main__":
    main() 