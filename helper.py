from PyPDF2 import PdfReader

def extract_pdf_text(file_path):
    pdf_file = PdfReader(file_path)
    text_data = ''
    for pg in pdf_file.pages:
        text_data += pg.extractText()
    return text_data
