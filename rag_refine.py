import os
import sys
import glob
import logging
import asyncio
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import httpx
import numpy as np
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_refine")

# Load environment variables
load_dotenv("/www/wwwroot/mcp_deepwiki/.env")


@dataclass
class Config:
    API_KEY: str = os.getenv("OPENAI_API_KEY")
    BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    LLM_MODEL: str = os.getenv("OPENAI_MODEL", "glm-4.7")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embedding-3")

    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 3


config = Config()

if not config.API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    sys.exit(1)


class EmbeddingClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.headers = {
            "Authorization": f"Bearer {config.API_KEY}",
            "Content-Type": "application/json"
        }
        self.base_url = config.BASE_URL.rstrip('/')

    async def get_embedding(self, text: str) -> List[float]:
        try:
            url = f"{self.base_url}/embeddings"
            payload = {
                "model": config.EMBEDDING_MODEL,
                "input": text
            }
            response = await self.client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []

    def _print_progress(self, label: str, idx: int, total: int):
        bar_len = 30
        filled_len = int(bar_len * idx / max(total, 1))
        bar = "█" * filled_len + "-" * (bar_len - filled_len)
        sys.stdout.write(f"\rEmbedding {label} [{bar}] {idx}/{total}")
        sys.stdout.flush()
        if idx == total:
            sys.stdout.write("\n")

    async def get_embeddings_batch(self, texts: List[str], label: str = "") -> List[List[float]]:
        embeddings = []
        total = len(texts)
        for idx, text in enumerate(texts, start=1):
            emb = await self.get_embedding(text)
            if emb:
                embeddings.append(emb)
            self._print_progress(label or "chunks", idx, total)
        return embeddings


class LLMClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=180.0)
        self.headers = {
            "Authorization": f"Bearer {config.API_KEY}",
            "Content-Type": "application/json"
        }
        self.base_url = config.BASE_URL.rstrip('/')

    async def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "temperature": temperature
            }
            response = await self.client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""


class VectorStore:
    def __init__(self):
        self.documents: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.sources: List[str] = []

    def add_documents(self, documents: List[str], vectors: List[List[float]], sources: List[str]):
        if not documents or not vectors:
            return

        self.documents.extend(documents)
        self.sources.extend(sources)

        new_vectors = np.array(vectors, dtype="float32")
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack((self.vectors, new_vectors))

    def search(self, query_vector: List[float], k: int = 3) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.vectors) == 0:
            return []

        query_vec = np.array(query_vector, dtype="float32")
        norm_query = np.linalg.norm(query_vec)
        norm_vectors = np.linalg.norm(self.vectors, axis=1)

        similarity = np.dot(self.vectors, query_vec) / (norm_vectors * norm_query + 1e-10)
        top_k_indices = np.argsort(similarity)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            results.append({
                "content": self.documents[idx],
                "source": self.sources[idx],
                "score": float(similarity[idx])
            })
        return results


class RAGRefiner:
    def __init__(self):
        self.embedder = EmbeddingClient()
        self.llm = LLMClient()
        self.vector_store = VectorStore()

    def chunk_text(self, text: str, source: str) -> tuple[List[str], List[str]]:
        chunks = []
        sources = []

        paragraphs = re.split(r"\n\n+", text)
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < config.CHUNK_SIZE:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    sources.append(source)
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())
            sources.append(source)

        return chunks, sources

    async def build_knowledge_base(self, repo_dir: str, exclude_file: str, selected_files: Optional[List[str]] = None):
        """Index selected markdown files (or all except the target if none selected)."""
        logger.info(f"Building knowledge base from {repo_dir}...")

        md_files = glob.glob(os.path.join(repo_dir, "*.md"))
        target_abs = os.path.abspath(exclude_file)

        # If selection provided, only keep those basenames
        selected_set = set(selected_files) if selected_files else None
        if selected_set is not None:
            logger.info(f"Embedding will run on {len(selected_set)} selected files.")
        else:
            logger.info("No selection provided; embedding all markdown files except target.")

        for file_path in md_files:
            if os.path.abspath(file_path) == target_abs:
                continue
            basename = os.path.basename(file_path)
            if selected_set is not None and basename not in selected_set:
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks, sources = self.chunk_text(content, basename)
            if chunks:
                embeddings = await self.embedder.get_embeddings_batch(chunks, label=basename)
                self.vector_store.add_documents(chunks, embeddings, sources)
                logger.info(f"Indexed {len(chunks)} chunks from {basename}")

    async def generate_title(self, repo_name: str, description: str = "", overview_content: str = "") -> str:
        """Generate a concise Chinese title for the repository"""
        logger.info("Generating AI title...")

        system_prompt = (
            "你是一个专业的技术文档标题撰写专家。"
            "请根据仓库名称、描述和概述内容，生成一个简洁的中文标题。\n"
            "要求：\n"
            "1. 标题长度在 10 个字以内\n"
            "2. 准确概括仓库的核心功能和价值\n"
            "3. 使用简洁的技术术语\n"
            "4. 只返回标题，不要有其他任何解释或标点符号"
        )

        user_content = f"仓库名称：{repo_name}\n"
        if description:
            user_content += f"描述：{description}\n"
        if overview_content:
            user_content += f"概述（前200字）：{overview_content[:200]}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        title = await self.llm.chat_completion(messages, temperature=0.2)
        # Clean up the title
        title = title.strip().strip('"').strip("'").strip("。").strip("：")
        logger.info(f"Generated title: {title}")
        return title

    async def generate_draft(self, content: str, readme_content: Optional[str] = None) -> str:
        """Phase 1: Generate Draft with RAG placeholders"""
        logger.info("Generating Draft...")

        system_prompt = (
            "你是一名专业的技术文档撰写者和软件架构师。"
            "你的任务是优化、重构和翻译GitHub项目的技术文档。"
            "输入是原始的文档页面，可能包含过多的实现细节、"
            "大量的源文件引用列表和非结构化的笔记。\n\n"
            "你的目标是：\n"
            "0. **项目简介**: 在文档的最前面，用一段话简要、清晰、重点明确地描述这个仓库的核心功能和价值。\n"
            "1. **翻译为中文**: 将所有说明性文本翻译为中文。保留技术术语（变量名、函数名、类名、文件路径）为英文。\n"
            "2. **去除噪音**: 移除'相关源文件'、'源代码引用'或类似不会增加架构理解的实现细节列表。只有在文件对解释至关重要时才引用。\n"
            "3. **重构与总结**: 重新组织内容，使其更具逻辑性和可读性。重点关注'为什么'和'如何工作'的高层次理解。突出关键功能和架构决策。\n"
            "4. **简洁性**: 合并冗余部分。让想要快速理解系统架构的开发者更容易消化。\n"
            "5. **识别盲区 (RAG)**: 对于文中提到的关键概念、架构模块或复杂机制，如果你觉得原文解释不够详细，"
            "或者你需要更多背景信息才能准确解释，请不要强行解释，而是插入一个RAG标记。\n"
            "6. **RAG标记格式**: `<!-- NEED_RAG: [搜索关键词] -->`，请确保关键词具体且有助于从其他文档中检索到相关信息。\n\n"
            "输出格式为干净、格式良好的Markdown，并遵循下面的注意事项。\n"
            "1. 不要包含mermaid图，将mermaid图转换为纯文本代码图表达。\n"
            "2. 不要包含任何表格，把原本用表格表达的内容转换为清晰的文本段落。\n"
            "3. 不要使用任何加粗语法，即** **。"
        )

        user_content = f"请处理以下文档：\n\n{content}"
        if readme_content:
            user_content += f"\n\n---\n\n以下是项目的 README 文件，可以帮助你更好地理解项目：\n\n{readme_content}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "system", "content": system_prompt}
        ]

        return await self.llm.chat_completion(messages)

    def extract_placeholders(self, draft: str) -> List[str]:
        # Improved regex to handle potential extra spaces and different formats
        return re.findall(r"<!--\s*NEED_RAG:\s*\[?(.*?)\]?\s*-->", draft)

    def get_surrounding_text(self, draft: str, placeholder_str: str, window: int = 800) -> str:
        """Return a snippet of the draft around the placeholder to give the LLM local context."""
        idx = draft.find(placeholder_str)
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(draft), idx + len(placeholder_str) + window)
        return draft[start:end]

    async def select_documents_for_rag(self, draft: str, candidate_files: List[str]) -> List[str]:
        placeholders = self.extract_placeholders(draft)
        if not placeholders:
            logger.info("No RAG placeholders found; skipping document selection.")
            return []

        file_list_text = "\n".join(candidate_files)
        placeholder_text = "\n".join(placeholders)

        system_prompt = (
            "你是一名技术文档分析专家。根据需要补充的关键词，"
            "请从给定的文件名列表中挑选最可能包含相关信息的文档。"
            "返回严格的JSON，键为 selected_documents（字符串数组），可选 reasoning 字段。"
        )

        user_content = (
            f"需要补充的关键词列表:\n{placeholder_text}\n\n"
            f"可用的文档文件名列表:\n{file_list_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        raw = await self.llm.chat_completion(messages)
        try:
            data = json.loads(raw)
            selected = data.get("selected_documents", []) if isinstance(data, dict) else []
            selected = [f for f in selected if f in candidate_files]
            logger.info(f"Selector chose {len(selected)} documents for embedding")
            return selected
        except Exception as e:
            logger.error(f"Failed to parse selector JSON: {e}; raw response: {raw}")
            return []

    async def expand_with_rag(self, draft: str) -> str:
        """
        Single LLM call: provide all placeholders, their contexts, and the full draft;
        ask the model to return the final Markdown with placeholders replaced.
        """
        logger.info("Expanding Draft with RAG (single-shot)...")

        placeholders = self.extract_placeholders(draft)
        if not placeholders:
            logger.info("No RAG placeholders found.")
            return draft

        # Collect contexts for each placeholder
        items = []
        for keyword in placeholders:
            logger.info(f"Retrieving context for: {keyword}")
            query_vec = await self.embedder.get_embedding(keyword)
            if not query_vec:
                logger.warning(f"No embedding for keyword: {keyword}")
                continue

            results = self.vector_store.search(query_vec, k=config.TOP_K)
            if not results:
                logger.warning(f"No retrieval results for keyword: {keyword}")
                continue

            context = "\n\n".join([f"--- Source: {r['source']} ---\n{r['content']}" for r in results])
            placeholder_str = f"<!-- NEED_RAG: [{keyword}] -->"
            draft_snippet = self.get_surrounding_text(draft, placeholder_str)

            items.append({
                "keyword": keyword,
                "placeholder": placeholder_str,
                "draft_snippet": draft_snippet,
                "context": context
            })

        if not items:
            logger.warning("No contexts retrieved; returning original draft.")
            return draft

        # Build a single prompt with all items and the full draft
        items_text = "\n\n".join(
            [
                f"- placeholder: {it['placeholder']}\n"
                f"  keyword: {it['keyword']}\n"
                f"  draft_snippet:\n{it['draft_snippet']}\n"
                f"  retrieved_context:\n{it['context']}"
                for it in items
            ]
        )

        system_prompt = (
            "你是一名专业的技术文档翻译和扩写专家。给你：\n"
            "1) 完整草稿（中文，含 RAG 占位符。RAG标记格式: `<!-- NEED_RAG: [搜索关键词] -->`），\n"
            "2) 每个占位符对应的检索上下文（可能包含英文）。\n\n"
            "任务：\n"
            "- 逐个替换占位符：阅读检索上下文，将其中的关键信息提取并**翻译为专业、详细的中文**，插入到原草稿对应的占位符位置。\n"
            "- **严禁输出英文说明**：即使上下文是英文，扩写内容也必须是高质量的中文技术说明。\n"
            "- 保持原草稿的结构、语言风格和未被替换的部分完全不变。\n"
            "- 输出最终完整的中文 Markdown 文档。"
        )

        user_prompt = (
            f"完整草稿（含占位符）：\n{draft}\n\n"
            f"占位符明细与上下文：\n{items_text}\n\n"
            "请输出最终完整的 Markdown 文档。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        final_md = await self.llm.chat_completion(messages)
        return final_md


async def main():
    output_root = "/www/wwwroot/mcp_deepwiki/output"
    
    # Find all Overview.md files in subdirectories of output
    all_overview_files = glob.glob(os.path.join(output_root, "**", "*Overview.md"), recursive=True)
    
    if not all_overview_files:
        # Fallback to any .md files if no Overview.md found
        all_overview_files = glob.glob(os.path.join(output_root, "**", "*.md"), recursive=True)
        
    if not all_overview_files:
        logger.error(f"No markdown files found in {output_root}")
        return

    # Filter out already refined files
    all_overview_files = [f for f in all_overview_files if "_Refined" not in f and "_draft" not in f]

    print("\nAvailable documents to refine:")
    for i, f in enumerate(all_overview_files):
        print(f"[{i}] {os.path.relpath(f, output_root)}")
    
    try:
        choice = input(f"\nSelect document index (default 0): ").strip()
        idx = int(choice) if choice else 0
        target_file = all_overview_files[idx]
    except (ValueError, IndexError):
        target_file = all_overview_files[0]

    base_dir = os.path.dirname(target_file)
    refined_dir = os.path.join(base_dir, "refined")
    os.makedirs(refined_dir, exist_ok=True)
    
    filename = os.path.basename(target_file)
    output_file = os.path.join(refined_dir, filename.replace(".md", "_Refined.md"))
    draft_file = os.path.join(refined_dir, filename.replace(".md", "_Refined_draft.md"))
    
    logger.info(f"Selected target: {target_file}")
    logger.info(f"Outputs will be saved to: {refined_dir}")

    refiner = RAGRefiner()

    # Read target file
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 1) Generate Draft
    draft = await refiner.generate_draft(content)
    with open(draft_file, "w", encoding="utf-8") as f:
        f.write(draft)
    logger.info(f"Draft saved to {draft_file}")

    # 2) LLM-based document selection
    candidate_files = [os.path.basename(p) for p in glob.glob(os.path.join(base_dir, "*.md")) if os.path.abspath(p) != os.path.abspath(target_file)]
    selected_files = await refiner.select_documents_for_rag(draft, candidate_files)

    # 3) Confirm before embedding
    if selected_files:
        logger.info(f"RAG placeholders found: {len(refiner.extract_placeholders(draft))}. Selected files for embedding: {len(selected_files)}")
    else:
        logger.info("No selected files for embedding (either no placeholders or selector returned empty).")
    proceed = input("Proceed to embedding? type 'yes' to continue: ").strip().lower()
    if proceed != "yes":
        logger.info("Embedding aborted by user.")
        return

    # 4) Build knowledge base only for selected files
    await refiner.build_knowledge_base(base_dir, target_file, selected_files)

    # 5) Expand with RAG
    final_content = await refiner.expand_with_rag(draft)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_content)
    logger.info(f"Final refined content saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
