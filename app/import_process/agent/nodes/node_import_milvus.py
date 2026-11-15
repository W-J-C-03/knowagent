import sys

from pymilvus.milvus_client import milvus_client

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.import_milvus_utils import step2_prepare_collection, step3_delete_old_data, step4_insert_collection
from app.utils.task_utils import add_running_task, add_done_task


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    # 1.进入结点的日志输出【节点 + 模型】记录任务状态【哪个任务开始了】-》给前端推送信息
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了。现在的状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        # 1.获取输入的数据并校验
        chunks = state.get('chunks')
        if not chunks:
            logger.error(f">>> [{function_name}]没有chunk数据，请检查")
            raise ValueError("没有chunk数据")
        # 2.没有集合，要创建集合 collection （filed，index，collection）
        milvus_client = step2_prepare_collection(state)
        # 3.删除旧数据
        step3_delete_old_data(milvus_client, chunks[0]['item_name'])
        # 4.插入chunks的数据即可
        chunks = step4_insert_collection(milvus_client, chunks)
        state['chunks'] = chunks
    except Exception as e:
        logger.error(f"Milvus切片数据入库节点执行失败：{str(e)}", exc_info=True)
        raise ValueError(f"Milvus切片数据入库节点执行失败：{str(e)}")
    finally:
        logger.info(f">>> [{function_name}]成功完成，现在状态为：{state}")
        add_done_task(state['task_id'], function_name)
    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")