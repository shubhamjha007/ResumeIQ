import fitz


def extract_text_from_pdf(pdf_file):
    """
    Extract text from every page of a PDF.

    Args:
        pdf_file: PDF file uploaded through Streamlit.

    Returns:
        Extracted text as a single string.
    """

    if pdf_file is None:
        return ""
    try:
        data = pdf_file.read()
        if not data:
            return ""
        with fitz.open(stream=data, filetype="pdf") as document:
            return "".join(page.get_text() for page in document).strip()
    except (fitz.FileDataError, RuntimeError, ValueError, AttributeError) as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc
