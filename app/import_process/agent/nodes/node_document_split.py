import os
import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.document_split_utils import step1_get_content_split, step2_split_by_title, step3_refine_chunks, \
    step4_backup_chunks
from app.utils.task_utils import add_running_task, add_done_task

"""
    完成对md内容切块
    最终：chunks -》存储块的集合 chunks -》 备份到本地 -》 chunks.json
    1.参数校验
    2.粗粒度切割语义完善 -》使用标题切割
    3.特殊场景，一个文档没有标题，给予默认标题
    4.细粒度切割md大小和重叠合适：大 -》再切 小-》合并
      大小合适，语义完整的chunks
    5.数据备份和chunks属性的修改
    返回state
"""
MAX_CONTENT_LENGTH = 2000
MIN_CHUNK_COUNT = 500
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了。现在的状态为: {state}")
    add_running_task(state['task_id'], function_name)
    try:
        # 1.参数校验：材料是否完整
        md_content, file_title = step1_get_content_split(state)
        # 2.粗粒度切割语义完善 -》使用标题切割
        sections, title_count, lines_count = step2_split_by_title(md_content, file_title)
        # 3.特殊场景，一个文档没有标题，给予默认标题
        if title_count == 0:
            # 没有标题
            sections = [{"title":"没有主题", "content":md_content, "file_title":file_title}]
        # 4.细粒度切割md大小和重叠合适：大 -》再切小 -》合并 大小合适，语义完整的chunks
        sections = step3_refine_chunks(sections, MAX_CONTENT_LENGTH, MIN_CHUNK_COUNT)
        # 5.数据备份和chunks属性的修改
        state['chunks'] = sections
        step4_backup_chunks(state, sections) # 备份
    except Exception as e:
        # 处理异常
        logger.error(f">>> [{function_name}]使用minerU解析发生异常，异常信息：{str(e)}")
        raise
    finally:
        logger.info(f">>> [{function_name}]成功完成，现在状态为：{state}")
        add_done_task(state['task_id'], function_name)
    return state

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")