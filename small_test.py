import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HOME'] = 'E:/data/huggingface'
os.environ['HF_HOME'] = ''
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import faiss

device =  "cuda" if torch.cuda.is_available() else "cpu"

# 加载文档并拆分
loader = TextLoader('data\documents\introduction.txt', encoding='utf-8')
documents = loader.load()

# 文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len, 
    separators=["\n\n", "-", "\n", "。", "？", "！", "(?<=\. )", " ", ""]
)

documents = text_splitter.split_documents(documents)
import numpy as np
print('Average Document Split Length: ', np.mean([len(doc.page_content) for doc in documents]))

# 使用本地模型嵌入
# model_name = "E:/data/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
model_name = 'E:/data/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d'


# 初始化HuggingFace嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# # 创建自定义FAISS索引
# d = 384  # MiniLM模型的维度
# nlist = 100  # 聚类中心数量
# quantizer = faiss.IndexFlatL2(d)
# index = faiss.IndexIVFFlat(quantizer, d, nlist)

# # 创建向量库
# vectorstore = FAISS(
#     embedding_function=embeddings.embed_query,
#     index=index
# )
# vectorstore.add_documents(documents)

# 转换为向量并存入FAISS数据库
# vectorstore = FAISS.from_documents(documents, embeddings) #, index_name='Flat')
vectorstore = Chroma.from_documents(documents, embeddings) #, index_name='Flat')

def get_prompt(query: str, vectorstore, top_k: int = 3):
    retrieved_docs = vectorstore.similarity_search(
                                                    query, 
                                                    k=top_k, 
                                                    search_type='similarity',   # 可选 "mmr" 最大边际相关
                                                    # search_type='mmr',
                                                    search_kwargs={"k": 30}     # 扩大候选池提高召回率
                                                )
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"你现在是一个直播助手，负责根据产品相关内容，在评论区回答问题。回复必须严格按照产品说明书，不能随意添加篡改内容，表达尽量精炼、平易近人。\n内容:{context}\n\n问题:{query}"
    return prompt


import requests

# 配置OpenRouter参数
OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# MODEL_NAME = "meta-llama/llama-3-70b-hf"  # 根据实际需求选择模型
# MODEL_NAME = "deepseek/deepseek-v3-base:free"
# MODEL_NAME = "google/gemini-2.5-pro-exp-03-25:free"
# MODEL_NAME = 'deepseek/deepseek-chat-v3-0324:free'

MODEL_NAME = 'qwen/qwen2.5-vl-32b-instruct:free'

# 转换生成参数
gen_params = {
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,  # 注意：部分模型可能不支持此参数
    "max_tokens": 1024,          # 参数名改为max_tokens
    # do_sample 参数不需要，由temperature自动控制
}

# 封装API请求函数
def openrouter_chat(prompt, history):
    # print('==\n', history)
    # messages = [{"role": ("user" if msg["role"] == "user" else "assistant"),
    #              "content": msg["content"]} 
    #              for msg in history]
    messages = history
    #### for chat-only model #####
    # messages.append({"role": "user", "content": prompt})
    #### for chat-and-vision model #####
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    # print('==\n', messages)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/jiafeng-yan",  # 需要替换为你的域名
        "X-Title": "RAG test"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        **gen_params
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    response = result['choices'][0]['message']['content']
    return response
    # return response, history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    # return result['choices'][0]['message']['content'], history + [
    #     {"role": "user", "content": prompt},
    #     {"role": "assistant", "content": result['choices'][0]['message']['content']}
    # ]

def get_response_from_openrouter(query: str, history: list, vectorstore, top_k: int = 3):
    prompt = get_prompt(query, vectorstore, top_k)

    print('\n' + ('-' * 40) + '\n' + prompt + '\n')
    # response, history = openrouter_chat(prompt, history)
    response = openrouter_chat(prompt, history)
    history.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
    print(f"{MODEL_NAME}: " + response + '\n')
    return response, history

# 使用示例
# print('你好')
# response, history = openrouter_chat("你好", history=[])
# print(response)



# # 使用本地LLM模型
# # 加载一个轻量级的本地模型，例如ChatGLM-6B或其他适合中文的小模型
# # local_model_name = "THUDM/chatglm2-6b-int4"  # 使用量化版模型节省资源
# local_model_name = r'E:/data/huggingface/modules/transformers_modules/Qwen/Qwen-1_8B-Chat'

# # 初始化tokenizer和模型
# tokenizer = AutoTokenizer.from_pretrained(local_model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     local_model_name,
#     trust_remote_code=True,
#     device_map={'': device},
#     low_cpu_mem_usage=True,  # 降低CPU内存使用
#     is_decoder=True,
# )

# model = model.eval()

# # 定义生成参数
# gen_params = {
#     "temperature": 0.7,      # 控制随机性
#     "top_p": 0.9,            # 控制词汇多样性
#     "repetition_penalty": 1.1, # 避免重复
#     "max_new_tokens": 512,   # 最大生成长度
#     "do_sample": True        # 启用采样
# }

# # 首次对话
# print('你好')
# response, history = model.chat(tokenizer, "你好", history=[], **gen_params)

# # history = []
# # while True:
# #     # 获取用户输入
# #     user_input = input("User: ")
# #     response, history = model.chat(tokenizer, user_input, history, **gen_params)
# #     print("AI: " + response)

# # print("AI: " + response)

# def get_response(query: str, history: list, vectorstore, top_k: int = 3):
#     prompt = get_prompt(query, vectorstore, top_k)

#     print('\n' + ('-' * 40) + '\n' + prompt + '\n')
#     response, history = model.chat(tokenizer, prompt, history, **gen_params)
#     print("Qwen-1.8B: " + response + '\n')
#     return response, history

if __name__ == "__main__":
    # 测试

    queries =[
        # '哇咔咔老年营养奶粉适合哪些年龄段的人群使用？',
        # '该奶粉含有哪些核心成分，对骨骼健康有什么帮助？',
        # '哇咔咔老年营养奶粉的正确食用方法是什么？',
        # '哇咔咔老年营养奶粉的生产厂家是哪家公司，在哪里生产？',
        # '哇咔咔老年营养奶粉的价格是多少，有什么规格可选？',
        '与市场上其他老年奶粉相比，哇咔咔老年营养奶粉的保质期是多久？',
        '有什么历史底蕴吗？',
        '有专家认证吗？',
        '通过国际标准了吗？',
    ]

    history = []
    for query in queries:
        # response, history = get_response(query, history, vectorstore, top_k=3)
        response, _ = get_response_from_openrouter(query, history, vectorstore, top_k=3)