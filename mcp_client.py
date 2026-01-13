import asyncio
import os
import re
import logging
from mcp import ClientSession
from mcp.client.sse import sse_client
from typing import List, Tuple

logger = logging.getLogger(__name__)

class DeepWikiMCPClient:
    def __init__(self, sse_url: str = "https://mcp.deepwiki.com/sse"):
        self.sse_url = sse_url

    def _save_to_markdown(self, repo_name: str, content_text: str) -> Tuple[str, List[str]]:
        base_dir = os.path.join(os.path.dirname(__file__), "output", repo_name.replace("/", "_"))
        os.makedirs(base_dir, exist_ok=True)
        
        raw_pages = re.split(r'^#\s+(?!#)(.*)$', content_text, flags=re.MULTILINE)
        saved_files = []
        
        intro_content = raw_pages[0].strip()
        if intro_content:
            file_path = os.path.join(base_dir, "00_Introduction.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(intro_content)
            saved_files.append(file_path)

        for i in range(1, len(raw_pages), 2):
            title = raw_pages[i].strip()
            content = raw_pages[i+1].strip() if i+1 < len(raw_pages) else ""
            if not title:
                continue
                
            index = (i // 2) + 1
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            filename = f"{index:02d}_{safe_title}.md"
            file_path = os.path.join(base_dir, filename)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{content}")
            
            saved_files.append(file_path)
        
        return base_dir, saved_files

    async def fetch_and_save(self, repo_name: str) -> str:
        """Fetch repo contents from MCP and save as markdown files."""
        logger.info(f"Connecting to DeepWiki MCP for: {repo_name}...")
        async with sse_client(self.sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                read_tool = next((t for t in tools_result.tools if "read_wiki_contents" in t.name), None)
                
                if not read_tool:
                    raise Exception("Tool 'read_wiki_contents' not found in MCP.")

                result = await session.call_tool(read_tool.name, arguments={"repoName": repo_name})
                
                full_content = ""
                for content in result.content:
                    if content.type == "text":
                        full_content += content.text
                
                if not full_content:
                    raise Exception(f"No content received for {repo_name}")

                output_dir, files = self._save_to_markdown(repo_name, full_content)
                logger.info(f"Saved {len(files)} pages to: {output_dir}")
                return output_dir
