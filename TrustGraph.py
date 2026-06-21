import json
import time
import os
import networkx as nx
import pandas as pd
from openai import OpenAI
from tqdm import tqdm  # 添加进度条

# ==== 配置项 ====
OPENAI_KEY = "sk-V9He0b5b0372b2316ba5786853240bc2d8b56c05c28fcEjA"
OPENAI_BASE = "https://api.gptsapi.net/v1"
INPUT_FILE = "../../Datasets/Weibo/filtered_weibo.json"
# INPUT_FILE = "../../Datasets/filtered_pheme.json"

MODEL = "gpt-4o-mini"

# ==== 初始化 OpenAI 客户端 ====
client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE)

# ==== 工具函数 ====
def load_json(path):
    return json.load(open(path, 'r', encoding='utf-8')) if os.path.exists(path) else {}

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_trust_scores_batch(news_content, comment_list, cache, raw_output, news_id, model=MODEL, retries=3):
    news_id_str = str(news_id)

    # 如果已经有 LLM 原始输出，直接解析并返回
    if news_id_str in raw_output:
        print(f"🔁 直接从缓存读取 LLM 输出: {news_id_str}")
        existing_result = raw_output[news_id_str]
        scores = {}

        if existing_result == "":
            cache[news_id_str] = {}
            return {}

        lines = [line for line in existing_result.strip().split("\n") if ":" in line]
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                cid, val = parts
                try:
                    idx = int(cid.strip().lower().replace("comment_", ""))
                    score = float(val.strip())
                    scores[str(idx)] = max(0.0, min(1.0, score))
                except:
                    continue

        cache[news_id_str] = scores
        return scores

    # 否则新调用 LLM
    if not comment_list:
        raw_output[news_id_str] = ""
        cache[news_id_str] = {}
        return {}

    comment_input = "\n".join([f"comment_{i+1}: {comment}" for i, comment in enumerate(comment_list)])
    prompt = f"""
You are a social media trust analysis expert. Based on the following news content and multiple user comments, evaluate how much each commenter trusts the news content.

Your task is to assign one score between 0 (not trusting at all) and 1 (fully trusting) for each comment. Higher scores mean more trust.

Special instructions:
- If the comment is a **retweet or repost**, assign a score of 0.9
- If the comment is a **"like" or contains [赞]**, assign a score of 0.7
- Otherwise, analyze the tone, meaning and semantics of the comment

News:
{news_content.strip()}

Now evaluate the following comments:
{comment_input}

Please return the result in this format:
comment_1: 0.90
comment_2: 0.10
comment_3: 0.70
...
""".strip()

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            result = response.choices[0].message.content.strip()
            raw_output[news_id_str] = result  # 保存原始响应

            scores = {}
            lines = [line for line in result.split("\n") if ":" in line]
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    cid, val = parts
                    try:
                        idx = int(cid.strip().lower().replace("comment_", ""))
                        score = float(val.strip())
                        scores[str(idx)] = max(0.0, min(1.0, score))
                    except:
                        continue

            cache[news_id_str] = scores
            return scores

        except Exception as e:
            print(f"[Retry {attempt+1}] LLM batch failed for news {news_id}: {e}")
            time.sleep(3)

    # fallback
    raw_output[news_id_str] = "FAILED"
    cache[news_id_str] = {str(i): 0.5 for i in range(len(comment_list))}
    return cache[news_id_str]


def build_trust_graph_from_json_with_llm(json_path, cache_path, llm_output_path):
    data = load_json(json_path)
    cache = load_json(cache_path)
    llm_output = load_json(llm_output_path)
    trust_graphs = {}
    edge_records = []

    for idx, (news_id, entry) in enumerate(tqdm(sorted(data.items(), key=lambda x: int(x[0])), desc="🚧 构建原始信任图"), start=1):
        news_id_str = str(news_id)
        content = entry.get("content", "")
        comments = entry.get("comments", [])
        if not isinstance(comments, list):
            comments = []

        comments = [c.strip() for c in comments if isinstance(c, str) and c.strip()][:10]

        # 🧠 尝试从缓存获取 scores；如无则调用 LLM
        if news_id_str in cache and news_id_str in llm_output:
            scores = cache[news_id_str]
        else:
            scores = get_trust_scores_batch(content, comments, cache, llm_output, news_id)

        # ✅ 图构建逻辑始终执行
        G = nx.DiGraph()
        news_node = f"news_{news_id}"
        G.add_node(news_node, type="news", content=content)

        for i, comment in enumerate(comments):
            user_node = f"user_{news_id}_{i}"
            G.add_node(user_node, type="user", comment=comment)
            weight = scores.get(str(i), 0.5)
            G.add_edge(user_node, news_node, weight=weight, comment=comment)

            edge_records.append({
                "news_id": news_id,
                "from": user_node,
                "to": news_node,
                "comment": comment,
                "weight": weight
            })

        trust_graphs[news_id_str] = G

        # 如果是新生成的，就更新缓存文件（可选）
        # ✅ 每条数据处理完都立即保存缓存，防止中断丢失
        save_json(dict(sorted(cache.items(), key=lambda x: int(x[0]))), cache_path)
        save_json(dict(sorted(llm_output.items(), key=lambda x: int(x[0]))), llm_output_path)

    edge_df = pd.DataFrame(edge_records)
    return trust_graphs, edge_df

