import asyncio
import sys
import logging
import os
import re
from mcp import ClientSession
from mcp.client.sse import sse_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_demo")

def save_to_markdown(repo_name, content_text):
    # Create base directory for the repo
    base_dir = os.path.join(os.path.dirname(__file__), "output", repo_name.replace("/", "_"))
    os.makedirs(base_dir, exist_ok=True)
    
    # Split content by Level 1 Headers (e.g., "# Header")
    # This regex looks for a line starting with a single '#' followed by a space
    raw_pages = re.split(r'^#\s+(?!#)(.*)$', content_text, flags=re.MULTILINE)
    
    saved_files = []
    
    # re.split with capturing group returns [non-matched, group1, non-matched, group1, ...]
    # Here: [text_before, title1, content1, title2, content2, ...]
    
    # Handle possible text before the first header
    intro_content = raw_pages[0].strip()
    if intro_content:
        file_path = os.path.join(base_dir, "00_Introduction.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(intro_content)
        saved_files.append(file_path)

    # Process title and content pairs
    for i in range(1, len(raw_pages), 2):
        title = raw_pages[i].strip()
        content = raw_pages[i+1].strip() if i+1 < len(raw_pages) else ""
        
        if not title:
            continue
            
        # Create a safe filename with a counter to maintain order
        index = (i // 2) + 1
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        filename = f"{index:02d}_{safe_title}.md"
        file_path = os.path.join(base_dir, filename)
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{content}")
        
        saved_files.append(file_path)
        print(f"📄 Saved: {filename}")
    
    return base_dir, saved_files

async def main():
    url = "https://mcp.deepwiki.com/sse"
    target_repo = "anthropics/claude-code"
    print(f"Connecting to {url}...")
    
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("\n✅ Connected to DeepWiki MCP!")
                
                tools_result = await session.list_tools()
                read_tool = next((t for t in tools_result.tools if "read_wiki_contents" in t.name), None)
                
                if not read_tool:
                    print("❌ Tool 'read_wiki_contents' not found.")
                    return

                print(f"\n🚀 Fetching contents for: {target_repo}...")
                
                # Call the tool with correct argument name 'repoName'
                result = await session.call_tool(read_tool.name, arguments={"repoName": target_repo})
                
                full_content = ""
                for content in result.content:
                    if content.type == "text":
                        full_content += content.text
                
                if full_content:
                    print("\n📂 Parsing and saving pages...")
                    output_dir, files = save_to_markdown(target_repo, full_content)
                    print(f"\n✨ Success! Saved {len(files)} pages to: {output_dir}")
                else:
                    print("⚠️ No content received from MCP.")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
