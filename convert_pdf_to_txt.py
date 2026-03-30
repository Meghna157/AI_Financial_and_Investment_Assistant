from pypdf import PdfReader
import os

PDF_DIR = "docs/pdf"   # Put all your PDF reports here
TXT_DIR = "docs/txt"   # TXT files will be saved here

os.makedirs(TXT_DIR, exist_ok=True)

def pdf_to_text(pdf_path, txt_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:       # Correct way to iterate pages
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

def convert_all_pdfs():
    for file in os.listdir(PDF_DIR):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, file)
            company = file.split("_")[0]
            txt_path = os.path.join(TXT_DIR, f"{company}_annual_report.txt")
            pdf_to_text(pdf_path, txt_path)
            print(f"Converted {file} -> {txt_path}")

if __name__ == "__main__":
    convert_all_pdfs()
