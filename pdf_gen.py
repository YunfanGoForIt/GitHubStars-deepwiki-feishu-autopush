import os
import sys
import glob
from md2pdf.converter import convert_markdown_to_pdf

def generate_pdfs():
    # Base output directory
    output_base = "/www/wwwroot/mcp_deepwiki/output"
    
    # Find all 02_Overview.md files in subdirectories
    overview_files = glob.glob(os.path.join(output_base, "*/02_Overview.md"))
    
    if not overview_files:
        print("No 02_Overview.md files found.")
        return

    for md_path in overview_files:
        repo_dir = os.path.dirname(md_path)
        pdf_name = "02_Overview.pdf"
        pdf_path = os.path.join(repo_dir, pdf_name)
        
        print(f"📄 Converting: {md_path} -> {pdf_path}")
        
        try:
            # md2pdf-mermaid provides a direct conversion function
            convert_markdown_to_pdf(
                markdown_text=open(md_path, 'r', encoding='utf-8').read(),
                output_path=pdf_path,
                title="Overview"
            )
            print(f"✅ Successfully generated: {pdf_path}")
        except Exception as e:
            print(f"❌ Failed to convert {md_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    generate_pdfs()
