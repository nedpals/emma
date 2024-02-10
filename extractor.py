pdf_path = "Handbook 2018.pdf"

def extract_content():
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(pdf_path).load()

