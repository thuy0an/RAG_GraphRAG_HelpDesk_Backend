from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from Features.LangChainAPI.LangChainDTO import ChatRequest

def System_Instruction(req: ChatRequest):
  return f"""
  ## AI Instructions

  ## Communication
  - **Response Language**: VIETNAMESE.

  Request: {req.message}
  """

ZERO_SHOT_PROMPT = f"""
  Tóm tắt nội dung sau: 

  Lịch sử manga bắt đầu từ rất sớm. Ở Nhật, người dân đã sớm có hứng thú với một loại nghệ thuật về tranh ảnh (sau này là manga) Manga thời kì này vẫn chỉ đơn giản là những dải truyện tranh ngắn. 
  Tuy vậy, giá trị giải trí của nó là điều không ai có thể phủ nhận. Không những thế, manga còn giữ một vị trí quan trọng và đầy quan trọng xuyên suốt lịch sử mỹ thuật Nhật Bản. 
  Manga phát triển từ ukiyo-e theo kiểu vẽ tranh. Nó phát triển nhanh chóng sau thế chiến thứ 2. Manga được hầu hết các hạng người đọc ở Nhật. 
  Do hầu hết các danh từ trong tiếng Nhật không ở dạng số nhiều nên manga có thể dùng để chỉ nhiều loại truyện tranh, đôi khi trong tiếng Anh cũng được viết là mangas
"""

FEW_SHOT_PROMPT = f"""
Tóm tắt nội dung sau: 

Hãy giúp tôi tóm tắt các văn bản dưới đây theo phong cách ngắn gọn, súc tích và giữ lại các thông tin cốt lõi nhất
Ví dụ 1:
Văn bản: 'Công nghệ 5G không chỉ mang lại tốc độ internet nhanh hơn gấp nhiều lần so với 4G, mà còn mở ra kỷ nguyên mới cho Internet vạn vật (IoT), xe tự lái và phẫu thuật từ xa nhờ độ trễ cực thấp.'
Tóm tắt: 5G giúp tăng tốc độ internet và thúc đẩy phát triển IoT, xe tự lái, y tế nhờ ưu điểm độ trễ thấp

Ví dụ 2:
Văn bản: 'Để bảo vệ sức khỏe tim mạch, các chuyên gia khuyến cáo mỗi người nên tập thể dục ít nhất 30 phút mỗi ngày, duy trì chế độ ăn ít muối và hạn chế các thực phẩm chứa chất béo bão hòa.'
Tóm tắt: Tập thể dục hằng ngày và ăn uống lành mạnh (ít muối, ít béo) là cách bảo vệ tim mạch hiệu quả

Nhiệm vụ của bạn:
Văn bản: 
Lịch sử manga bắt đầu từ rất sớm. Ở Nhật, người dân đã sớm có hứng thú với một loại nghệ thuật về tranh ảnh (sau này là manga) Manga thời kì này vẫn chỉ đơn giản là những dải truyện tranh ngắn. 
Tuy vậy, giá trị giải trí của nó là điều không ai có thể phủ nhận. Không những thế, manga còn giữ một vị trí quan trọng và đầy quan trọng xuyên suốt lịch sử mỹ thuật Nhật Bản. 
Manga phát triển từ ukiyo-e theo kiểu vẽ tranh. Nó phát triển nhanh chóng sau thế chiến thứ 2. Manga được hầu hết các hạng người đọc ở Nhật. 
Do hầu hết các danh từ trong tiếng Nhật không ở dạng số nhiều nên manga có thể dùng để chỉ nhiều loại truyện tranh, đôi khi trong tiếng Anh cũng được viết là mangas

Tóm tắt:
"""

COT_PROMPT = f"""
Hãy tóm tắt văn bản theo phương pháp suy luận từng bước. Đầu tiên, hãy xác định các ý chính, sau đó loại bỏ chi tiết phụ và cuối cùng là tổng hợp thành một câu tóm tắt

Ví dụ:
Văn bản: 'Việc áp dụng làm việc từ xa giúp doanh nghiệp tiết kiệm 30% chi phí vận hành văn phòng, 
đồng thời giúp nhân viên giảm căng thẳng do không phải di chuyển trong giờ cao điểm, từ đó tăng năng suất lao động lên đáng kể.'

Suy nghĩ:
Ý chính 1: Doanh nghiệp tiết kiệm 30% chi phí.
Ý chính 2: Nhân viên giảm căng thẳng và tăng năng suất.
Mối quan hệ: Làm việc từ xa mang lại lợi ích cho cả doanh nghiệp và nhân viên.

Tóm tắt: Làm việc từ xa giúp tối ưu chi phí cho doanh nghiệp và nâng cao hiệu suất cũng như tinh thần cho nhân viên

Nhiệm vụ của bạn:

Văn bản:             
Lịch sử manga bắt đầu từ rất sớm. Ở Nhật, người dân đã sớm có hứng thú với một loại nghệ thuật về tranh ảnh (sau này là manga) Manga thời kì này vẫn chỉ đơn giản là những dải truyện tranh ngắn. 
Tuy vậy, giá trị giải trí của nó là điều không ai có thể phủ nhận. Không những thế, manga còn giữ một vị trí quan trọng và đầy quan trọng xuyên suốt lịch sử mỹ thuật Nhật Bản. 
Manga phát triển từ ukiyo-e theo kiểu vẽ tranh. Nó phát triển nhanh chóng sau thế chiến thứ 2. Manga được hầu hết các hạng người đọc ở Nhật. 
Do hầu hết các danh từ trong tiếng Nhật không ở dạng số nhiều nên manga có thể dùng để chỉ nhiều loại truyện tranh, đôi khi trong tiếng Anh cũng được viết là mangas

Suy nghĩ: (Hãy phân tích từng bước tại đây)

Tóm tắt:
"""

REACT_PROMPT= f"""
## Role
Bạn là một tác giả viết lách có nhiều kinh nghiệm viết blog

## Input
Nhiệm vụ của bạn là viết một blog về "Cảm xúc của tôi về Fate Stay Night" khoảng 300 từ

## Workflow
Đối với [INPUT] luôn thực hiện theo chu kỳ chặt chẽ:
Suy nghĩ (thought) -> Hành động (Action) -> Quan sát (Observation)
Sau chu kì các hành động đó đến bước Tổng kết (Summary), bước này bạn phải thực nhiệm vụ

## 
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
