"""实体关系抽取模块 — 基于 spaCy NER + 共现关系"""

import spacy
from typing import Optional


class EntityExtractor:
    """
    基于 spaCy en_core_web_sm 的命名实体识别与同句共现关系抽取。

    提取的实体类型：PERSON, ORG, GPE（地缘政治实体）, LOC, PRODUCT, EVENT, WORK_OF_ART
    关系类型：默认 cooccurrence（同一句中出现）
    """

    # 我们关心的实体类型
    VALID_TYPES = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"}

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

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

        doc = self.nlp(text)

        # 提取有效实体
        entities = []
        seen = set()
        for ent in doc.ents:
            if ent.label_ not in self.VALID_TYPES:
                continue
            name = ent.text.strip()
            if len(name) < 2:
                continue
            key = (name.lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            entities.append({
                "name": name,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "text_span": ent.text,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
            })

        # 同句共现关系
        relations = []
        for sent in doc.sents:
            sent_ents = [
                {"name": e.text.strip(), "type": e.label_}
                for e in sent.ents
                if e.label_ in self.VALID_TYPES and len(e.text.strip()) >= 2
            ]
            # 去重
            sent_ents = list({(e["name"].lower(), e["type"]): e for e in sent_ents}.values())

            for i in range(len(sent_ents)):
                for j in range(i + 1, len(sent_ents)):
                    relations.append({
                        "source": sent_ents[i]["name"],
                        "source_type": sent_ents[i]["type"],
                        "target": sent_ents[j]["name"],
                        "target_type": sent_ents[j]["type"],
                        "type": "cooccurrence",
                        "confidence": 1.0,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                    })

        return entities, relations
