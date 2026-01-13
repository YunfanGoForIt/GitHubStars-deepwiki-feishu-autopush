import asyncio
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

class DeepWikiIndexer:
    def __init__(self, email: str):
        self.email = email
        self.base_url = "https://deepwiki.com"

    async def request_indexing(self, repo_full_name: str) -> bool:
        """
        Trigger indexing request for unindexed repositories using Next.js Server Actions.
        """
        logger.info(f"Triggering indexing request for {repo_full_name} via direct API...")
        
        # DeepWiki uses Next.js Server Actions. 
        # The endpoint for server actions is the page URL itself with specific headers.
        target_url = f"{self.base_url}/{repo_full_name}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept": "text/x-component",
            "Next-Router-State-Tree": "%5B%5B%22%22%2C%7B%22children%22%3A%5B%22org%22%2C%7B%22children%22%3A%5B%22repo%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%5D%7D%5D%7D%5D%7D%2Cnull%2Cnull%2Ctrue%5D%5D",
            "Next-Action": "76d1e370a2f3a6937e2978a3f8595a864d309068", # This is the action ID for indexing
            "Content-Type": "text/plain;charset=UTF-8",
            "Origin": self.base_url,
            "Referer": target_url,
        }

        # Payload format for Next.js Server Actions (typically JSON-like array)
        # For indexing, it usually takes [email, repo_full_name]
        payload = f'["{self.email}","{repo_full_name}"]'

        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30.0) as client:
            try:
                # 1. First check if already indexed
                check_resp = await client.get(target_url)
                if check_resp.status_code == 200 and ("Overview" in check_resp.text or "Table of Contents" in check_resp.text):
                    logger.info(f"Repository {repo_full_name} is already indexed.")
                    return True

                # 2. POST to trigger indexing action
                response = await client.post(target_url, content=payload)
                
                if response.status_code in [200, 303]:
                    logger.info(f"Indexing action triggered for {repo_full_name}. Status: {response.status_code}")
                    return True
                else:
                    logger.error(f"Failed to trigger indexing for {repo_full_name}. Status: {response.status_code}, Body: {response.text[:200]}")
                    return False

            except Exception as e:
                logger.error(f"Error during indexing request: {e}")
                return False

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    email = os.getenv("USER_EMAIL", "test@example.com")
    indexer = DeepWikiIndexer(email)
    asyncio.run(indexer.request_indexing("dai-hongtao/InkTime"))
