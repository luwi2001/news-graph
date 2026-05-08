"""文本段落切分模块"""

from typing import Optional


class TextChunker:
    """
    按近似 token 数切分长文本。

    由于 all-MiniLM-L6-v2 的 WordPiece tokenizer 中，
    英文 1 token ≈ 0.75 词 ≈ 4-5 字符，所以:
    - chunk_size=200 字符 ≈ 256 tokens（模型上限）
    - overlap=50 字符保证语义连续性
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # 确保 overlap 小于 chunk_size 的一半
        if self.overlap >= self.chunk_size // 2:
            self.overlap = self.chunk_size // 4

    def chunk(self, title: str, description: Optional[str] = None) -> list[str]:
        """
        将新闻切成段落列表。
        第一段始终包含 title + description 开头。
        """
        text = title.strip()
        if description:
            text = f"{text}: {description.strip()}"

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            # 尝试在句子边界（.!?）处切断，最多回退 30 字符
            if end < len(text):
                for punct in ['. ', '! ', '? ', '; ', '\n']:
                    pos = text.rfind(punct, start, end)
                    if pos != -1 and pos > start + self.chunk_size // 2:
                        end = pos + len(punct)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            start = end - self.overlap
            if start <= 0 or start >= len(text):
                break

        return chunks
