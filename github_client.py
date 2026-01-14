import os
import httpx
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GitHubConfig:
    token: str
    api_url: str = "https://api.github.com/user/starred"

class GitHubMonitor:
    def __init__(self, token: str):
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {token}",
        }
        self.api_url = "https://api.github.com/user/starred"

    async def fetch_recent_stars(self, limit: int = 30):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.api_url, 
                    headers=self.headers, 
                    params={"per_page": limit, "sort": "created", "direction": "desc"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching recent stars: {str(e)}")
            return []

    async def fetch_all_stars(self):
        all_stars = []
        page = 1
        per_page = 100
        
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    logger.info(f"Fetching stars page {page}...")
                    response = await client.get(
                        self.api_url,
                        headers=self.headers,
                        params={"per_page": per_page, "page": page, "sort": "created", "direction": "desc"}
                    )
                    response.raise_for_status()
                    stars = response.json()
                    
                    if not stars:
                        break
                        
                    all_stars.extend(stars)
                    
                    if len(stars) < per_page:
                        break
                        
                    page += 1
                except Exception as e:
                    logger.error(f"Error fetching all stars page {page}: {str(e)}")
                    break

        return all_stars

    async def fetch_repo_readme(self, repo_name: str) -> Optional[str]:
        """
        Fetch README content from a GitHub repository.
        Returns None if README doesn't exist or on error.
        """
        try:
            async with httpx.AsyncClient() as client:
                # Try common README filenames
                for readme_name in ["README.md", "README.zh.md", "README.zh-CN.md"]:
                    url = f"https://api.github.com/repos/{repo_name}/contents/{readme_name}"
                    response = await client.get(url, headers=self.headers)

                    if response.status_code == 200:
                        data = response.json()
                        # GitHub API returns base64 encoded content
                        import base64
                        content = base64.b64decode(data["content"]).decode("utf-8")
                        logger.info(f"✅ 成功获取 README: {repo_name}/{readme_name}")
                        return content
                    elif response.status_code == 404:
                        continue
                    else:
                        logger.warning(f"⚠️ 获取 README 失败 ({readme_name}): {response.status_code}")

                logger.info(f"ℹ️ 未找到 README: {repo_name}")
                return None
        except Exception as e:
            logger.warning(f"⚠️ 拉取 README 异常 [{repo_name}]: {str(e)}")
            return None
