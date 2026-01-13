import os
import sys
import logging
import asyncio
import httpx
import glob
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("refine_content")

# Load environment variables
load_dotenv("/www/wwwroot/mcp_deepwiki/.env")

class RefinerSettings:
    ZHIPU_API_KEY = os.getenv("OPENAI_API_KEY")
    ZHIPU_BASE_URL = os.getenv("OPENAI_BASE_URL")
    ZHIPU_MODEL = "glm-4.5-flash"  # Using same model as translator_service.py

settings = RefinerSettings()

class ContentRefiner:
    def __init__(self):
        self.api_key = settings.ZHIPU_API_KEY
        self.base_url = settings.ZHIPU_BASE_URL
        self.model = settings.ZHIPU_MODEL
        
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            sys.exit(1)

    async def process_content(self, content: str) -> str:
        """
        Refine the content: Translate to Chinese, remove source links, restructure, and summarize.
        """
        if not content:
            return ""

        logger.info(f"Processing content chunk (length: {len(content)})...")

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
            "4. **简洁性**: 合并冗余部分。让想要快速理解系统架构的开发者更容易消化。\n\n"
            "输出格式为干净、格式良好的Markdown，不要包含mermaid图，将mermaid图转换为纯文本代码图表达。"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "temperature": 0.3
        }

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                # Handle potential trailing slash in base_url
                base = self.base_url.rstrip('/') if self.base_url else "https://open.bigmodel.cn/api/paas/v4"
                url = f"{base}/chat/completions"
                
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"API Error {response.status_code}: {response.text}")
                    return content  # Return original on error to avoid data loss
                
                result = response.json()
                refined_text = result["choices"][0]["message"]["content"]
                return refined_text.strip()
                
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return content

async def main():
    # Target specific directory
    target_dir = "/www/wwwroot/mcp_deepwiki/output/facebook_react"
    output_dir = os.path.join(target_dir, "refined")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all markdown files
    # Only target 02_Overview.md as requested
    target_file = os.path.join(target_dir, "02_Overview.md")
    if os.path.exists(target_file):
        md_files = [target_file]
    else:
        logger.warning(f"02_Overview.md not found in {target_dir}")
        md_files = []
    
    if not md_files:
        logger.warning(f"No matching Markdown files found in {target_dir}")
        return

    refiner = ContentRefiner()
    
    for file_path in md_files:
        filename = os.path.basename(file_path)
        logger.info(f"Refining {filename}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Refine content
        refined_content = await refiner.process_content(content)
        
        # Save to output directory
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(refined_content)
            
        logger.info(f"Saved refined content to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
