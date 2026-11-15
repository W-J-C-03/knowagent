import os
import sys

from app.clients.milvus_utils import get_milvus_client
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.item_name_recognition_utils import step1_get_chunks, step2_build_context, step3_call_llm, \
    step4_update_chunks_and_state, step5_generate_embedding, step6_save_to_vector_db
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.escape_milvus_string_utils import escape_milvus_string
"""
    注解：这一步调用向量生成模型是为了：让大模型可以识别用户的问题
    目标：
        1.录用文本大模型识别当前chunks对应的item_name：用于区分不同的文档
        2.使用嵌入式模型（本地），将item_name生成向量存储到向量数据库
        3.修改state[chunks] -》chunk{ title parent_title part file_title content item_name -》每个赋值}
    实现步骤：
        1.校验和取值（file_title，chunks）
        2.构建上下文环境 chunks -》top 5 -》拼接成context文本
        3.调用大模型，拼接提示词，识别chunks对应item_name
        4.修改state_chunks -》item_name
        5.item_name生成向量（稠密/稀疏）
        6.存储向量到向量数据库 kb_item_name（id / file_title / item_name / 稠密和稀疏）
"""
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了。现在的状态为: {state}")
    add_running_task(state['task_id'], function_name)
    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    try:
        # 1.检验和取值 file_title，chunks
        chunks, file_title = step1_get_chunks(state)
        # 2.构建上下文环境 chunks -》top 5 -》拼接成context文本
        context = step2_build_context(chunks)
        # 3.调用大模型，拼接提示词，识别chunks对应item_name
        item_name = step3_call_llm(context, file_title)
        # 4.修改state_chunks -》item_name
        step4_update_chunks_and_state(state, item_name, chunks)
        # 5.item_name生成向量（稠密 / 稀疏）
        dense_vector, sparse_vector = step5_generate_embedding(item_name)
        # 6.存储向量到向量数据库kb_item_name（id / file_title / item_name / 稠密和稀疏）
        step6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)

    except Exception as e:
        # 处理异常
        logger.error(f">>> [{function_name}]使用minerU解析发生异常，异常信息：{str(e)}")
        raise
    finally:
        logger.info(f">>> [{function_name}]成功完成，现在状态为：{state}")
        add_done_task(state['task_id'], function_name)

    return state

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()