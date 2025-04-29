import os
import glob
import shutil
import json

from pydantic import BaseModel
from pdf2image import convert_from_path

from llm import get_vision_response

class PageSegmentSchema(BaseModel):
    text_segment: str
    context: str
    page_number: int

class PageSegmentsSchema(BaseModel):
    segments: list[PageSegmentSchema]

def extract_content(pdf_path="./handbook.pdf"):
    pages_dir = "./pages"
    extracted_dir = "./extracted_2"  # Store all pages in a common directory

    if os.path.exists(pages_dir):
        shutil.rmtree(pages_dir)

    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(extracted_dir, exist_ok=True)

    extracted: list[tuple[int, str, str]] = []  # (page_number, context, text)

    # Extract content from the PDF using pdf2image
    print("Extracting content from PDF...")
    convert_from_path(pdf_path, fmt="jpeg", output_folder=pages_dir)

    # Get all jpeg files in pages directory and convert to absolute paths
    image_paths = [os.path.abspath(p) for p in sorted(glob.glob(os.path.join(pages_dir, "*.jpg")))]

    ocr_prompt = """
You are an expert at extracting meaningful text segments from images to be used in a Retrieval-Augmented Generation (RAG) system. Your task is to process the provided image and output a structured response containing a concise, self-contained text segment and the page number (if present).  The goal is to create chunks that can be independently understood by a language model for answering questions. Follow these specific instructions carefully:

1. **Content Extraction & Chunking:** Extract all readable text from the image. Focus on extracting *complete* sentences or phrases that represent a single idea or concept. Do not attempt to stitch together incomplete fragments unless they form a coherent unit. Prioritize accuracy and clarity over attempting to "guess" missing words due to OCR errors.
2. **Contextual Information (Crucial for RAG):**  Before the extracted text, include a brief description of what the segment *is* within the larger document. Examples:
    * "Section Heading:" followed by the heading text
    * "Table Caption:" followed by the table caption
    * "Paragraph from Chapter 2:" (if applicable)
    * "Definition of [term]:" if defining a term
    * If no clear context is present, use "Text Segment:"
3. **Page Number Identification & Formatting:**
   * **If a page number is visible in the footer,** extract it and format the footer as either: `'[page_number] | 2023 Edition'` OR `'2023 Edition | [page_number]'`.  Use whichever order appears in the image. Replace `[page_number]` with the actual numerical page number.
   * **If no page number is visible,** set the page number to -1.
4. **Markdown Conversion (Minimal):** Convert the extracted text into Markdown format *only* as needed for basic readability and structural clarity.  Avoid complex formatting like tables or elaborate lists unless they are essential to understanding the content.  Focus on:
    * **Headings:** Identify headings and convert them appropriately (e.g., `# Heading 1`, `## Heading 2`).
    * **Paragraphs:** Separate paragraphs with blank lines.
5. **Incomplete Content Handling:**
   * **If the content appears to be cut off or incomplete,** append `[content_incomplete]` to the *end* of the extracted text segment. Do *not* add this tag if it's a cover page or intentionally truncated material.  This is to indicate potential missing information for RAG purposes.
6. **Output Format:** Your output should be in the following JSON format:

   ```json
   {
     "text_segment": "[Extracted text segment with Markdown formatting]",
     "context": "[Contextual description of the segment]",
     "page_number": [integer]
   }
   ```
""".strip()
    for idx, image_path in enumerate(image_paths):
        print(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")
        page_number = idx + 1

        # Check if page cache exists
        page_cache_file = os.path.join(extracted_dir, f"page_{idx}.json")
        if os.path.exists(page_cache_file):
            with open(page_cache_file, 'r') as f:
                cached_data = json.load(f)
                if cached_data and "segments" in cached_data:
                    print(f"Using cached data for page {idx}")
                    for segment in cached_data["segments"]:
                        extracted.append((
                            segment["page_number"], 
                            segment["context"], 
                            segment["text_segment"]
                        ))

                if len(extracted) >= len(image_paths):
                    print("All pages processed, skipping further processing.")
                    break
                continue

        # Generate the response using the LLM
        response = get_vision_response(image_path, ocr_prompt, response_format=PageSegmentsSchema, temperature=0.45)
        
        segments = []
        if isinstance(response, PageSegmentsSchema):
            segments = response.segments
        elif isinstance(response, dict) and "segments" in response:
            segments = response["segments"]
        elif isinstance(response, (dict, str)):
            # Handle single segment response for backward compatibility
            text_segment = response.get("text_segment", str(response)) if isinstance(response, dict) else str(response)
            context = response.get("context", "Text Segment:") if isinstance(response, dict) else "Text Segment:"
            segments = [PageSegmentSchema(text_segment=text_segment, context=context, page_number=page_number)]

        page_segments = []
        for segment in segments:
            if isinstance(segment, dict):
                text_segment = segment.get("text_segment", "")
                context = segment.get("context", "")
            else:
                text_segment = segment.text_segment
                context = segment.context

            print(f"Processing segment (page_number: {page_number}): {text_segment[:100]}...")
            
            if text_segment.endswith("[content_incomplete]"):
                text_segment = text_segment[:-len("[content_incomplete]")]
            
            extracted.append((page_number, context, text_segment))
            page_segments.append({
                "page_number": page_number,
                "context": context,
                "text_segment": text_segment
            })

        # Cache the page results
        cached_content = {"segments": page_segments}
        with open(page_cache_file, 'w') as f:
            json.dump(cached_content, f)

    return extracted
