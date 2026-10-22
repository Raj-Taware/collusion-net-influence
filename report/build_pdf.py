"""Render report/report.md to report/report.pdf (Markdown -> HTML -> headless Edge/Chrome print).

    python report/build_pdf.py
"""

import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: "Times New Roman", Georgia, serif; font-size: 11pt; line-height: 1.4; color: #111; }
h1 { font-size: 17pt; text-align: center; margin-bottom: 4pt; }
h1 + p, h1 + p + p { text-align: center; margin: 2pt 0; }
h2 { font-size: 13pt; margin-top: 16pt; }
h3 { font-size: 11.5pt; }
table { border-collapse: collapse; margin: 8pt auto; font-size: 9.5pt; }
th, td { border-top: 1px solid #999; border-bottom: 1px solid #999; padding: 3pt 6pt; text-align: left; }
img { display: block; max-width: 100%; margin: 8pt auto; }
code { font-size: 9.5pt; }
"""


def main():
    md = (HERE / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables"])
    html = HERE / "report.html"  # kept next to report.md so ../results/ image paths resolve
    html.write_text(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{body}", encoding="utf-8")
    browser = next(b for b in BROWSERS if Path(b).exists())
    pdf = HERE / "report.pdf"
    profile = HERE / ".pdf-profile"  # separate profile so a running browser window does not swallow the job
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, timeout=120)
    html.unlink()
    print("wrote", pdf)


if __name__ == "__main__":
    main()
