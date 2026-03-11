from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def get_oscar_nominations_2026(category: str) -> str:
    """Trả về danh sách đề cử Oscar 2026 theo từng hạng mục cụ thể."""
    
    nominations = {
        "Best Picture": "Sinners, Căn nhà ký ức (Home), Anora, Sentimental Value, Flow...",
        "Best Actor": "Leonardo DiCaprio (Sinners), Ethan Hawke, ...",
        "Best Actress": "Karla Sofía Gascón (Emilia Pérez), Mikey Madison (Anora), ...",
        "Best Director": "Sean Baker (Anora), Ryan Coogler (Sinners), ..."
    }

    result = nominations.get(category, "Không tìm thấy dữ liệu cho hạng mục này.")
    
    return f"Đề cử cho hạng mục {category}: {result}"

class AgentService:
    def __init__(self, provider, callbacks):
        self.provider = provider
        self.callbacks = callbacks or {}
        pass

    async def search_oscar(self):
        agent = create_agent(
            model=self.provider,
            tools=[get_oscar_nominations_2026],
            system_prompt="Bạn là trợ lý AI chuyên về giải thưởng Oscar. Hãy trả lời chính xác và đầy đủ."
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Đề cử cho hạng mục Best Picture?"}]}
        )

        return result.get("messages", [])[-1].content

    