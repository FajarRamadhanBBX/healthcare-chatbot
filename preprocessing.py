from langchain.schema import Document

def filter_documents(documents: list[Document]) -> list[Document]:
    """
    Filter documents to include only those between "Introduction" (as a header) and "References".
    """
    capturing = False
    filtered_documents = []

    for doc in documents:
        text = doc.page_content
        lower_text = text.lower()
        lines = lower_text.splitlines()  # Pisahkan teks menjadi baris-baris

        captured_lines = []  # Tempat untuk menyimpan baris yang difilter

        for line in lines:
            # Periksa apakah baris adalah header "Introduction"
            if line.strip() == "introduction":
                capturing = True
                continue

            # Periksa apakah baris adalah header "References"
            if capturing and "references" in line and len(line) < 20:
                capturing = False
                break

            # Tambahkan baris ke daftar jika dalam mode capturing
            if capturing:
                captured_lines.append(line)

        # Gabungkan kembali baris-baris yang difilter menjadi satu string
        if captured_lines:
            filtered_text = "\n".join(captured_lines)
            filtered_documents.append(Document(page_content=filtered_text, metadata=doc.metadata))
    
    return filtered_documents