import sys
from app.utils.task_utils import *
from dotenv import load_dotenv
from app.lm.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
load_dotenv()

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5


def step1_merge_rrf_mcp(state):
    """
    进行rrf+mcp数据整合
    :param state:
    :return:
    """
    # 1.state 获取不同路的数据
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    # 2.准备一个列表容器
    chunks_list = []
    # 3.循环进行数据添加
    # 3.1 local rrf
    for chunk in rrf_chunks:
        # chunk :{id, distance, entity}
        entity = chunk.get("entity") or chunk
        chunk_id = entity.get("chunk_id")
        content = entity.get("content")
        title = entity.get("title")
        chunks_list.append({"chunk_id": chunk_id, "text": content, "title": title, "source":"local", "url":""})
    # 3.2 web mcp
    for doc in web_search_docs:
        text = doc.get("snippet")
        url = doc.get("url")
        title = doc.get("title")
        chunks_list.append({"chunk_id": "", "text": text, "title": title, "source":"web", "url": url})
    logger.info(f"多路数据融合，最终结果为：{chunks_list}")
    return chunks_list


def step2_rerank_doc_list(doc_list, state):
    """
    使用rerank模型进行精排（模型给答案打分）
    :param doc_list:
    :param state:
    :return:
    """
    # 1.获取原有的问题
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    # 2.获取问题对应的所有答案
    text_list = [doc['text'] for doc in doc_list]
    # 3.加载rerank模型 BAAI 北京智源
    rerank = get_reranker_model()
    # 4.处理数据 设置 问题+答案，存于列表中（这个格式为重排序模型所识别的），让模型打分
    # [ [问题，答案]，[问题，答案]]，问题＋答案 大小为模型所支持的，512
    question_pairs = [[rewritten_query, text] for text in text_list]
    # normalize = True 默认为false，分数范围不确定，差别大
    scores = rerank.compute_score(question_pairs, normalize=True)
    # 5.将原有的数据添加对应的分
    doc_list_with_score = []
    for score, item in zip(scores, doc_list):
        item['score'] = score
        doc_list_with_score.append(item)
    doc_list_with_score.sort(key=lambda x: x.get("score", 0), reverse=True)
    logger.info(f"已经完成排序和打分，最终结果为：{doc_list_with_score}")
    return doc_list_with_score

def step3_topk_and_gap(rerank_score_list):
    """
    对rerank模型打分以后的有序集合进行再次算法筛选
    取出topk个元素
    :param rerank_score_list:
    :return:
    [
        {
            text: 内容 snippet content,
            chunk_id: chunk_id rrf有 | mcp None
            title: title
            url: rrf None | mcp url
            source: web ->mcp  || local -> rrf
            score: rerank打分
        }
    ]
    """
    max_topk = RERANK_MAX_TOPK
    min_topk = RERANK_MIN_TOPK
    gap_abs = RERANK_GAP_ABS
    gap_ratio = RERANK_GAP_RATIO
    # 1.思考最大截取数量
    # top不应该大于列表长度
    topk = min(max_topk, len(rerank_score_list))
    # 2.循环处理数据列表，进行双指针处理和比较
    if topk > min_topk:
        # 正常循环 min-1, topk-1
        for index in range(min_topk-1, topk-1):
            # 双指针
            score1 = rerank_score_list[index].get("score", 0.0)
            score2 = rerank_score_list[index+1].get("score", 0.0)
            # 算分差
            gap = score1 - score2
            rel = gap / (abs(score1)+1e-6)
            if gap >=gap_abs or rel >= gap_ratio:
                logger.info(f"数据集合{index}和{index+1}存在分数断崖")
                topk = index + 1
                break
    # else:
        # min_topk = topk 不用管
        # min_topk > topk list 0
    # 3.截取确定的数量topk
    topk_doc_list = rerank_score_list[:topk]
    # 4.打印日志处理
    logger.info(f"最终截取的长度：{topk}")
    # 5.返回结果
    return topk_doc_list




def node_rerank(state):
    """
    节点作用：rrf + mcp -> 精排 rerank模型 -> chunk -> 调用大模型打分 -> 算法 -> topk
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    print("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1.非同源路的结果合并（rrf + mcp）放到一个集合中
    """
        rrf = {id:chunk_id, distance: 0.x, entity: {chunk_id, content, title}}
        mcp = {snippet: 内容, title: 标题, url: 关联的文章或图片的地址}
        {
            text: 内容 snippet content,
            chunk_id: chunk_id rrf有，mcp None
            title: title
            url: rrf None, mcp url
            source: web ->mcp  || local -> rrf
            
        }
    """
    doc_list = step1_merge_rrf_mcp(state)
    # 2.启用rerank进行精排 （数据和分）
    # doc_list 包含了从向量数据库查询的结果，和从web上搜索的内容
    rerank_score_list = step2_rerank_doc_list(doc_list, state)
    # 3.启动算法进行防断崖以及topk处理
    reranked_docs = step3_topk_and_gap(rerank_score_list)
    # 4.结果装到state中即可

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"reranked_docs": reranked_docs}

if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> 启动 node_rerank 本地测试")
    print("="*50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8},
        {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1} # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"} # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？", # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("="*50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
