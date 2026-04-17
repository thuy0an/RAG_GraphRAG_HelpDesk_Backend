from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from typing import Dict, List
from Features.LangChainAPI.LangChainDTO import ChatRequest

def System_Instruction(req: ChatRequest):
  return f"""
  ## AI Instructions

  ## Communication
  - **Response Language**: VIETNAMESE. If the user writes in English, respond in English.
  - Do NOT use any language other than Vietnamese or English.

  Request: {req.message}
  """

# ====================
# STRUCTED OUTPUT
# ====================
YOUTUBE_TEMPLATE="""
Trích xuất thông tin video YouTube từ đoạn mô tả sau.

Mô tả:
{description}
"""

YOUTUBE_DESCRIPTION = {
    "description": """
    Video "Learn FastAPI in 30 Minutes" được đăng bởi kênh CodeMaster.
    Video đạt 125000 lượt xem sau 2 ngày.
    Đây không phải là YouTube Shorts.
    Ngày đăng: January 10, 2025.
    """
}

# ====================
# CONVERSATIONAL RAG
# ====================

# Mapping từ role_key sang role string trong message objects
_ROLE_KEY_TO_ROLE = {
    "rag_content": "assistant_rag",
    "graphrag_content": "assistant_graphrag",
}


def format_history_block(
    turns: List[Dict],
    role_key: str = "rag_content",
) -> str:
    """
    Format danh sách messages (từ MemoryRepository.get_recent_messages) thành
    ConversationHistory_Block để inject vào prompt.

    Args:
        turns: List[Dict] từ MemoryRepository._rows_to_messages().
               Mỗi dict có keys: role, content, turn_id, timestamp.
               Thứ tự: oldest first (đã được MemoryRepository đảm bảo).
        role_key: "rag_content" cho PaCRAG → filter role "assistant_rag"
                  "graphrag_content" cho GraphRAG → filter role "assistant_graphrag"

    Returns:
        str: ConversationHistory_Block đã format, hoặc "" nếu không có turns hợp lệ.
    """
    if not turns:
        return ""

    assistant_role = _ROLE_KEY_TO_ROLE.get(role_key, "assistant_rag")

    # Group messages theo turn_id (giữ thứ tự oldest first)
    turn_map: Dict[str, Dict] = {}
    turn_order: List[str] = []

    for msg in turns:
        tid = msg.get("turn_id")
        if not tid:
            continue
        if tid not in turn_map:
            turn_map[tid] = {"user": None, "assistant": None}
            turn_order.append(tid)

        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "user" and content:
            turn_map[tid]["user"] = content
        elif role == assistant_role and content:
            turn_map[tid]["assistant"] = content

    # Build block — chỉ include turns có đủ cả user và assistant
    lines = []
    for tid in turn_order:
        t = turn_map[tid]
        if t["user"] and t["assistant"]:
            lines.append(f"Người dùng: {t['user']}")
            lines.append(f"Trợ lý: {t['assistant']}")
            lines.append("")  # dòng trống ngăn cách giữa các turns

    if not lines:
        return ""

    # Bỏ dòng trống cuối cùng
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def PaC_template(context: str) -> str:
    """
    Template chuẩn cho PaCRAG retrieval (stateless — không có history).
    Dùng cho retrieve_full() và compare endpoints.
    Giữ {query} như placeholder để dùng với ChatPromptTemplate.
    """
    return f"""Bạn là trợ lý AI chuyên nghiệp. Trả lời câu hỏi dựa trên ngữ cảnh bên dưới.
Trả lời bằng tiếng Việt. Nếu câu hỏi bằng tiếng Anh, trả lời bằng tiếng Anh.
Không sử dụng bất kỳ ngôn ngữ nào khác.

Ngữ cảnh:
{context}

Câu hỏi: {{query}}

Hướng dẫn:
- Nếu câu hỏi là lời chào: chỉ chào lại, không dùng ngữ cảnh.
- Nếu là câu hỏi thực sự: trả lời đầy đủ dựa trên ngữ cảnh, sau đó thêm nguồn ở cuối theo định dạng:
  - Nguồn: <tên file>
  - Trang: <số trang>
- Nếu không có thông tin: trả lời "Tôi không có thông tin về vấn đề này, vui lòng liên hệ bộ phận hỗ trợ."

Trả lời:"""


def PaC_template_with_history(context: str, history_block: str = "") -> str:
    """
    Template PaCRAG có hỗ trợ conversation history.
    Khi history_block không rỗng, inject section lịch sử trước câu hỏi.
    Giữ {query} như placeholder để dùng với ChatPromptTemplate.

    Args:
        context: Formatted context từ _format_context_PaC().
        history_block: ConversationHistory_Block từ format_history_block().
                       Nếu rỗng, section lịch sử bị bỏ qua hoàn toàn.
    """
    # Escape curly braces trong history_block để tránh lỗi ChatPromptTemplate
    safe_history = history_block.replace("{", "{{").replace("}", "}}")

    history_section = ""
    if safe_history.strip():
        history_section = f"""
=== Lịch sử hội thoại ===
{safe_history}
=== Kết thúc lịch sử ===
"""

    return f"""Bạn là trợ lý AI chuyên nghiệp. Trả lời câu hỏi dựa trên ngữ cảnh bên dưới.
Trả lời bằng tiếng Việt. Nếu câu hỏi bằng tiếng Anh, trả lời bằng tiếng Anh.
Không sử dụng bất kỳ ngôn ngữ nào khác.

Ngữ cảnh:
{context}
{history_section}
Câu hỏi: {{query}}

Hướng dẫn:
- Nếu câu hỏi là lời chào: chỉ chào lại, không dùng ngữ cảnh.
- Nếu là câu hỏi tiếp nối (follow-up): sử dụng lịch sử hội thoại để hiểu ngữ cảnh.
- Nếu là câu hỏi thực sự: trả lời đầy đủ dựa trên ngữ cảnh, ưu tiên thông tin từ tài liệu hơn lịch sử hội thoại, sau đó thêm nguồn ở cuối theo định dạng:
  - Nguồn: <tên file>
  - Trang: <số trang>
- Nếu không có thông tin: trả lời "Tôi không có thông tin về vấn đề này, vui lòng liên hệ bộ phận hỗ trợ."

Trả lời:"""
