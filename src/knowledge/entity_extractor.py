"""实体关系抽取模块 — 基于 GLiNER"""

from typing import Optional
import re


class EntityExtractor:
    """
    基于 GLiNER 的命名实体识别与同句共现关系抽取。

    提取的实体类型：Person, Organization, Location, Product, Event, Policy, Industry
    关系类型：默认 cooccurrence（同一句中出现）
    """

    # GLiNER 使用的标签
    LABELS = [
        "Person",
        "Organization",
        "Location",
        "Product",
        "Event",
        "Policy",
        "Industry",
    ]

    # spaCy 类型到 GLiNER 类型的映射（向后兼容）
    TYPE_MAPPING = {
        "Person": "PERSON",
        "Organization": "ORG",
        "Location": "GPE",
        "Product": "PRODUCT",
        "Event": "EVENT",
        "Policy": "EVENT",
        "Industry": "ORG",
    }

    def __init__(self, model: str = "urchade/gliner_small-v2.1", threshold: float = 0.3):
        """
        Args:
            model: GLiNER 模型名称或路径
            threshold: 置信度阈值（默认 0.3，比默认 0.5 低以提升召回）
        """
        from gliner import GLiNER
        self.model = GLiNER.from_pretrained(model)
        self.threshold = threshold

    def _clean_text(self, text: str) -> str:
        """预处理文本：去掉多余空白"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _split_text(self, text: str, max_chars: int = 800) -> list[tuple[int, str]]:
        """
        将长文本按句子切分成段，每段不超过 max_chars。
        返回 (offset, segment) 列表。
        """
        # 按句子切分（简单按 .!? 切分）
        sentences = re.split(r'(?<=[.!?])\s+', text)
        segments = []
        current_segment = ""
        current_offset = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current_segment) + len(sent) + 1 > max_chars and current_segment:
                segments.append((current_offset, current_segment.strip()))
                current_offset += len(current_segment) + 1
                current_segment = sent
            else:
                if current_segment:
                    current_segment += " " + sent
                else:
                    current_segment = sent

        if current_segment:
            segments.append((current_offset, current_segment.strip()))

        return segments

    def extract(self, text: str, doc_id: int, chunk_id: Optional[int] = None):
        """
        从文本中提取实体和共现关系。

        Args:
            text: 待分析的文本
            doc_id: 所属文档 ID
            chunk_id: 所属 chunk ID（可选）

        Returns:
            (entities, relations)
            entities: list[dict] — {name, type, start, end, text_span}
            relations: list[dict] — {source, target, type, confidence}
        """
        if not text or len(text.strip()) < 10:
            return [], []

        text = self._clean_text(text)

        # 分段处理长文本
        segments = self._split_text(text, max_chars=800)

        all_entities = []
        for offset, segment in segments:
            if not segment:
                continue
            try:
                gliner_ents = self.model.predict_entities(
                    segment,
                    self.LABELS,
                    threshold=self.threshold,
                )
                for ent in gliner_ents:
                    # 计算在原始文本中的位置
                    start = offset + segment.find(ent['text'])
                    end = start + len(ent['text'])
                    all_entities.append({
                        "name": ent['text'].strip(),
                        "type": ent['label'],
                        "start": start,
                        "end": end,
                        "text_span": ent['text'],
                        "score": ent.get('score', 1.0),
                    })
            except Exception as e:
                # GLiNER 失败时静默跳过
                print(f"[WARN] GLiNER failed for doc {doc_id}: {e}")
                continue

        # 去重（按 name + type）
        seen = set()
        entities = []
        for ent in all_entities:
            # 映射到 spaCy 兼容类型
            mapped_type = self.TYPE_MAPPING.get(ent["type"], ent["type"])
            key = (ent["name"].lower(), mapped_type)
            if key in seen:
                continue
            seen.add(key)
            entities.append({
                "name": ent["name"],
                "type": mapped_type,
                "start": ent["start"],
                "end": ent["end"],
                "text_span": ent["text_span"],
                "doc_id": doc_id,
                "chunk_id": chunk_id,
            })

        # 同句共现关系
        relations = self._extract_relations(text, entities, doc_id, chunk_id)

        return entities, relations

    def _extract_relations(self, text: str, entities: list, doc_id: int, chunk_id: Optional[int] = None):
        """
        基于同句共现抽取实体关系。
        """
        if len(entities) < 2:
            return []

        # 按句子分组
        sentences = re.split(r'(?<=[.!?])\s+', text)
        relations = []
        seen_relations = set()

        for sent in sentences:
            sent_start = text.find(sent)
            sent_end = sent_start + len(sent)

            # 找出在当前句子中的实体
            sent_ents = []
            for ent in entities:
                # 实体与句子有重叠
                if ent["start"] < sent_end and ent["end"] > sent_start:
                    sent_ents.append(ent)

            # 去重
            unique_ents = []
            seen_names = set()
            for ent in sent_ents:
                key = (ent["name"].lower(), ent["type"])
                if key not in seen_names:
                    seen_names.add(key)
                    unique_ents.append(ent)

            # 生成共现关系
            for i in range(len(unique_ents)):
                for j in range(i + 1, len(unique_ents)):
                    src = unique_ents[i]
                    tgt = unique_ents[j]
                    rel_key = (src["name"].lower(), tgt["name"].lower())
                    if rel_key in seen_relations:
                        continue
                    seen_relations.add(rel_key)
                    seen_relations.add((tgt["name"].lower(), src["name"].lower()))

                    relations.append({
                        "source": src["name"],
                        "source_type": src["type"],
                        "target": tgt["name"],
                        "target_type": tgt["type"],
                        "type": "cooccurrence",
                        "confidence": 1.0,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                    })

        return relations
