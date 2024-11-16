import os
import pdfplumber
import json

def extract_text_from_pdf(pdf_path):
    pdf_text = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            paragraphs = text.split('\n')  # Separar en párrafos por saltos de línea
            pdf_text[page_num] = paragraphs
    return pdf_text

def save_pdf_text(pdf_text, output_path):
    with open(output_path, 'w') as f:
        json.dump(pdf_text, f, indent=4)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio del script
    pdf_path = os.path.join(base_dir, '../data/swebok-v3.pdf')  # Ruta relativa al script
    output_path = os.path.join(base_dir, '../data/pdf_text.json')

    pdf_text = extract_text_from_pdf(pdf_path)
    save_pdf_text(pdf_text, output_path)
    print(f"Texto extraído y guardado en {output_path}")
