from pypdf import PdfReader
import string

with open("english.txt", "r") as f:
    STOPWORDS = set(line.strip() for line in f)

def get_text(pdf_path):
    pdf = PdfReader(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(pdf.pages):
        cleaned_text = text_prep(page.extract_text(layout_mode='loose'))
        text_by_page.append((page_num, cleaned_text))
    return text_by_page

def text_prep(text, remove_whitespace=True, remove_punctuation=True, remove_stopwords=True):
    text = text.lower()

    if remove_whitespace:
        text = text.strip()
        text = text.replace("\n", " ")

    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    if remove_stopwords:
        words = text.split()
        filtered_words = []
        for word in words:
            if word not in STOPWORDS:
                filtered_words.append(word)
        text = " ".join(filtered_words)

    return text

def split_chunks(text, chunk_size=50, overlap=0):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks