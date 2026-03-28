import io
from PIL import Image
import fitz


def process_file_types(files: list):
    """
    Converts various input files (PIL Images, Streamlit uploads, PDF paths) into a 
    standardized multipart format for uploading or processing.
    """
    multipart_files = []
    for i, f in enumerate(files):
        # PIL Image
        if isinstance(f, Image.Image):
            buf = io.BytesIO()
            f.save(buf, format="PNG")
            buf.seek(0)
            multipart_files.append(("files", (f"screenshot_{i}.png", buf, "image/png")))

        # Streamlit UploadedFile
        elif hasattr(f, "type") and hasattr(f, "read") and hasattr(f, "name"):
            f_bytes = f.read()
            if f.type.startswith("image") or f.type == "application/pdf":
                mime = f.type
                fname = f.name
            else:
                raise ValueError(f"Unsupported file type: {f.name} ({f.type})")
            multipart_files.append(("files", (fname, io.BytesIO(f_bytes), mime)))

        # PDF path as string
        elif isinstance(f, str) and f.lower().endswith(".pdf"):
            with open(f, "rb") as pdf_file:
                data = pdf_file.read()
                multipart_files.append(
                    ("files", (f"document_{i}.pdf", io.BytesIO(data), "application/pdf"))
                )
        else:
            raise ValueError(f"Unsupported file type: {f}")
    return multipart_files



def extract_texts(ocr, files: list):
    """
    Extracts text from image and PDF files using OCR, returning a list 
    of dictionaries with filenames and their combined text content.
    """
    results = []

    for f in files:
        if f.content_type == "image/png":
            img_bytes = f.file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            ocr_res = ocr(img)
            res = [[t for b, t, crop in line] for line in ocr_res]
            results.append({"name": f.filename, "text": ''.join(item for sublist in res for item in sublist)})
        elif f.content_type == "application/pdf":
            pdf_bytes = f.file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pdf_text = ""

            def ocr_bytes(img_bytes):
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                ocr_res = ocr(img)
                res = [[t for b, t, crop in line] for line in ocr_res]
                return ''.join(item for sublist in res for item in sublist)

            for page in doc:
                blocks = sorted(page.get_text("dict")["blocks"], key=lambda b: b["bbox"][1])

                image_map = {
                    xref: doc.extract_image(xref)["image"]
                    for xref, *_ in page.get_images(full=True)
                    if xref
                }

                for b in blocks:
                    if b["type"] == 0:  # text
                        pdf_text += "".join(
                            span.get("text", "")
                            for line in b.get("lines", [])
                            for span in line.get("spans", [])
                        ) + "\n"

                    elif b["type"] == 1:  # image
                        xref = b.get("xref")
                        imgs = [image_map[xref]] if xref in image_map else image_map.values()
                        for img_bytes in imgs:
                            try:
                                pdf_text += ocr_bytes(img_bytes) + "\n"
                            except Exception:
                                pass

            results.append({"name": f.filename, "text": pdf_text})

        f.file.seek(0)

    return results