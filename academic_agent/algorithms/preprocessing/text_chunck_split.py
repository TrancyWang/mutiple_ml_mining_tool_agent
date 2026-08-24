import re
import logging
import re
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_to_paragraphs(text):
    # 以双换行、markdown标题、横线等方式切分段落
    paragraphs = re.split(r'(\n\s*\n|^#{1,6}\s+|^-{3,}|^={3,})', text, flags=re.MULTILINE)
    # 过滤空内容，合并正文
    result = []
    buffer = ""
    for part in paragraphs:
        part = part.strip()
        if part:
            if buffer:
                result.append(buffer)
                buffer = ""
            result.append(part)
    if buffer:
        result.append(buffer)
    return [p for p in result if p.strip()]

def split_to_sentences(text):
    # 支持中英文句子标点切分
    pattern = r'(?<=[。！？.!?])'
    return [s.strip() for s in re.split(pattern, text) if s.strip()]

def split_to_clauses(sentence):
    # 以逗号、分号等为子句切分点
    pattern = r'(?<=[,，;；])'
    return [c.strip() for c in re.split(pattern, sentence) if c.strip()]

def ragflow_chunk_split(
    text,
    max_length=350,
    min_length=150,
    stride=100
):
    """
    ragflow复杂chunk分割：段落-句子-子句三级切分，智能拼接补全，滑动窗口重叠。
    """
    paragraphs = split_to_paragraphs(text)
    raw_chunks = []

    for para in paragraphs:
        # 优先保留段落整体
        if len(para) <= max_length:
            raw_chunks.append(para)
            continue

        # 段落过长则按句子切分
        sentences = split_to_sentences(para)
        chunk = ''
        for sent in sentences:
            if len(chunk) + len(sent) < max_length:
                chunk += sent
            else:
                if len(chunk) < min_length:
                    # 如果chunk还太短，考虑按子句切分
                    clauses = split_to_clauses(sent)
                    for clause in clauses:
                        if len(chunk) + len(clause) < max_length:
                            chunk += clause
                        else:
                            raw_chunks.append(chunk)
                            chunk = clause
                else:
                    raw_chunks.append(chunk)
                    chunk = sent
        if chunk:
            raw_chunks.append(chunk)

    # 拼接小块，保证每个chunk长度合适
    merged_chunks = []
    buffer = ''
    for c in raw_chunks:
        if len(buffer) + len(c) < min_length:
            buffer += ('\n' + c) if buffer else c
        else:
            if buffer:
                merged_chunks.append(buffer)
            buffer = c
    if buffer:
        merged_chunks.append(buffer)

    # 滑动窗口，提升检索覆盖率
    final_chunks = []
    idx = 0
    while idx < len(merged_chunks):
        cur = merged_chunks[idx]
        if idx > 0 and stride < len(cur):
            prev = merged_chunks[idx-1]
            # 重叠部分拼接
            overlap = prev[-stride:] if len(prev) > stride else prev
            chunk = overlap + cur
            final_chunks.append(chunk)
        else:
            final_chunks.append(cur)
        idx += 1

    return final_chunks




def split_review_to_semantic_units(text):
    """
    这个是专门写论文给评句子使用的
    评论 → 语义单元（句子 / 子句级）
    每个单元 = 一个评价观点
    """
    # 先按句号、问号、感叹号切
    sentences = re.split(r'(?<=[。！？.!?])', text)

    units = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # 再按逗号、分号切子观点
        clauses = re.split(r'(?<=[,，;；])', sent)
        for c in clauses:
            c = c.strip()
            if len(c) >= 2:   # 防止空碎片
                units.append(c)

    return units

# ============ 使用示例 =============
if __name__ == "__main__":
    demo_text = """
# 标题1

这是一个测试段落，内容很丰富。支持中英文分句！This is an English sentence. 还有更多的内容，超长句子自动拆分，保证不会硬切割。  
让我们看看chunk能否完整保留段落语义。如果一段很长很长很长，需要进一步细分为合适的片段。

---

## 标题2
这里还有第二段。逗号、分号、甚至英文都能切分；This is a new test, with a lot of, commas, to split. 继续写点内容，测试chunk效果。
"""
    #
    # chunks = ragflow_chunk_split(demo_text, max_length=100, min_length=40, stride=30)
    # for i, chunk in enumerate(chunks):
    #     logger.info(f'=== Chunk {i+1} ===')
    #     logger.info(chunk)
    #     logger.info('------')

res = split_review_to_semantic_units(demo_text)
for unit in res:
    logger.info(unit)
    logger.info('------')