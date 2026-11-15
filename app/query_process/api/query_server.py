from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger, PROJECT_ROOT
from app.query_process.agent.state import create_default_state
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app

# 后续导入启动图对象
#from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service",description="掌柜智库查询服务！")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康状态
@app.get("/health")
async def health():
    logger.info(f"触发后台检测监察接口，数据一切正常")
    return {"status": "ok"}

# 返回chat.html
@app.get("/chat.html")
async def chat_html():
    # 查找chat.html页面的地址
    chat_html_path = PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat.html文件不存在")
    return FileResponse(chat_html_path)

# 发起提问接口
# 接收参数的类型
class QueryRequest(BaseModel):
    query: str = Field(..., title="查询内容")
    session_id: str = Field(None, title="会话id")
    is_stream: bool = Field(False, title="是否流式返回结果")

def run_query_graph(query: str, session_id: str, is_stream: bool):
    # 调用main_graph执行
    update_task_status(session_id, "processing", is_stream)
    state = create_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )
    try:
        final_state = query_app.invoke(state)
        answer = final_state.get("answer") or ""
        image_urls = final_state.get("image_urls") or []
        set_task_result(session_id, "answer", answer)
        update_task_status(session_id, "completed", is_stream)
        # 本次任务开启了，is_stream = True 把结果加入到队列中，sse可以取到
        if is_stream:
            # 唯一的终止信号：成功失败，有没有图片都要发，前端收到后close掉EventSource
            push_to_session(session_id, SSEEvent.FINAL, {
                "answer": answer,
                "status": "complete",
                "image_urls": image_urls
            })
            push_to_session(session_id, SSEEvent.CLOSE, {})
    except Exception as e:
        logger.exception(f"---session_id={session_id}，查询出现异常---{str(e)}")
        set_task_result(session_id, "error", str(e))
        update_task_status(session_id, "failed", is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.FINAL, {
                "answer": "",
                "status": "failed",
                "error": str(e),
                "image_urls": []
            })
            push_to_session(session_id, SSEEvent.CLOSE, {})  # ✅ 异常路径：也要发关闭信号

@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    background_tasks: BackgroundTasks 是 FastAPI 提供的异步后台任务机制，核心作用是：
    接口立刻返回响应给前端，同时把耗时任务放到后台继续执行。因为langgraph执行图的流程比较慢，但前端需要实时响应结果
    :param request:请求参数
    :param background_tasks:异步执行函数 is_stream = True
    :return:
    """
    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    # 判断是不是流式处理 （异步 -》先返回一个结果 开始处理 | 后台运行图，结果向前推进）
    if is_stream:
        # 只要开启流式处理，业务中就是将数据插入到队列中，{session_id, queue, [update_task_state, add_running_task, add_done_list]}
        # 创建当前session_id对应的队列 -》_session_stream
        create_sse_queue(session_id)
        # 异步执行 立即返回结果前端 || 中间的过程 sse一点一点给前端
        # 本次任务开启了 is_stream = True 把结果加入到队列 sse可以取到
        update_task_status(session_id, "processing", is_stream)
        background_tasks.add_task(run_query_graph, query, session_id, is_stream)
        logger.info(f"query:{query}以及开启了异步和流式处理")
        return {
            "session_id": session_id,
            "message": "本次查询处理中..."
        }
    else:
        # 同步执行
        await run_in_threadpool(run_query_graph, query, session_id, is_stream)
        # 获取最后一个节点插入的结果 node_answer_output (answer)
        answer = get_task_result(session_id, "answer")
        # 返回对应的json数据即可
        logger.info(f"query:{query}开启同步处理，处理结果为：{answer}")
        return {
            "answer": answer,
            "session_id": session_id,
            "message": "本次查询处理完毕",
            "done_list": []
        }

@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """

    :param session_id:
    :param request: 前端的原生请求对象，可以判断是否断开连接
    :return:
    """
    logger.info(f"session_id:{session_id}客户端，已经与后台建立了长连接")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )

@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 10):
    """
    :param session_id:
    :param limit: 切割的数量
    :return:
    """
    # 获取历史对话
    chats = get_recent_messages(session_id, limit)
    logger.info(f"查询历史对话，session_id = {session_id}成功，查询数据为：{chats}")
    return {
        "session_id": session_id,
        "items": chats
    }

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    # 删除历史对话
    delete_count = clear_history(session_id)
    logger.info(f"删除历史对话：session_id = {session_id}成功，删除了{delete_count}条数据")
    return {
        "deleted_count": delete_count,
        "message": f"{session_id}聊天记录删除成功"
    }


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """获取当前任务的状态"""
    # 假设 task_utils 里有个函数能获取状态，如果没有，你可以直接去查那个全局字典
    # 根据你的代码逻辑，update_task_status 应该是维护在某个地方
    # 这里假设有一个 get_task_status 函数返回 "processing", "completed" 等

    return {
        "session_id": session_id,
        "status": get_task_status(session_id) or "pending",
        "done_list": get_done_task_list(session_id),
        "running_list": get_running_task_list(session_id),
        "answer": get_task_result(session_id, "answer"),
        "error": get_task_result(session_id, "error"),
    }

@app.get("/")
async def index():
    """根路径直接跳到聊天页，避免打开 http://127.0.0.1:8001/ 看到 404"""
    return RedirectResponse(url="/chat.html")


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8001)

