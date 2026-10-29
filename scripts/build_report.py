"""Build docs/report.pdf from docs/report.md.

Markdown -> HTML (python-markdown, with tables) -> PDF via headless Chrome. Images are embedded as
data URIs so the PDF is self-contained.

Usage: python scripts/build_report.py
"""
import base64
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

DOCS = Path("docs")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.4;
       color: #111; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 20pt; margin-bottom: 4px; } h2 { font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #999; }
h3 { font-size: 11.5pt; } p { margin: 6px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #eee; }
img { max-width: 100%; max-height: 118mm; width: auto; display: block; margin: 8px auto 2px; page-break-inside: avoid; }
img + em, p > em:only-child { display: block; text-align: center; font-size: 9pt; color: #444; }
code { font-family: Menlo, monospace; font-size: 9pt; background: #f3f3f3; padding: 0 2px; }
li { margin: 2px 0; }
p:has(img) { page-break-inside: avoid; }
@page { size: A4; margin: 16mm 14mm; }
"""


def embed_images(html):
    def repl(m):
        src = Path("docs") / m.group(1)
        data = base64.b64encode(src.read_bytes()).decode()
        return f'src="data:image/jpeg;base64,{data}"'
    return re.sub(r'src="([^"]+\.jpg)"', repl, html)


def main():
    text = (DOCS / "report.md").read_text()
    # A photo number such as "#17" at the start of a wrapped line would read as a heading.
    text = re.sub(r"(?m)^#(?=\d)", r"\\#", text)
    body = markdown.markdown(text, extensions=["tables"])
    # Put each image's alt text under it as a caption.
    body = re.sub(r'<p><img alt="([^"]*)" (src="[^"]+") /></p>', r'<p><img alt="\1" \2 /><em>\1</em></p>', body)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{embed_images(body)}</body></html>"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html)
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={(DOCS / 'report.pdf').resolve()}", page.resolve().as_uri()],
                       check=True, capture_output=True)
    print("wrote", DOCS / "report.pdf", f"{(DOCS / 'report.pdf').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
