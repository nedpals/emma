import os
import glob
import shutil
import json

from pydantic import BaseModel
from pdf2image import convert_from_path

from llm import provider

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
**You are an AI expert specializing in processing educational documents, specifically school handbooks, for Retrieval-Augmented Generation (RAG) systems.** Your primary task is to analyze the provided image (a page from a school handbook) and extract meaningful, self-contained text chunks suitable for answering user questions accurately.

**Follow these instructions meticulously:**

1.  **Goal:** Create text chunks that represent distinct pieces of information (e.g., a specific rule, a policy section, contact details, a definition) that can be understood independently by a language model.

2.  **Chunking Strategy & Content Extraction:**
    *   Identify logical units of information on the page. This could be:
        *   A paragraph or multiple closely related paragraphs under a specific heading.
        *   A distinct rule or policy statement.
        *   A definition or explanation of a term.
        *   A list of items (try to keep related list items together within a chunk, potentially under their introductory sentence or heading).
        *   Contact information blocks.
    *   Extract all readable text within the identified logical unit.
    *   Prioritize extracting *complete* sentences or coherent phrases.
    *   **Accuracy is paramount:** Transcribe text exactly as it appears. Do *not* guess or paraphrase missing words due to OCR errors. If a word is clearly garbled or unreadable, you may represent it with `[unreadable]` or omit it if it severely breaks coherence. Do *not* invent content.
    *   Avoid splitting a single sentence, list item, or very tightly coupled idea across different chunks *if they appear together visually on the page*.

3.  **Contextual Description (Crucial for RAG):**
    *   For each chunk, provide a brief `context` string.
    *   This context should ideally identify the section or topic the chunk belongs to, using headings or subheadings visible on the page if possible. Examples: `"Context: Dress Code Policy - Shirts"`, `"Context: Attendance Procedures - Reporting Absences"`, `"Context: Academic Honesty - Plagiarism Definition"`, `"Context: School Contact Information - Main Office"`.
    *   If no clear heading is associated, describe the general topic (e.g., `"Context: General School Rules"`).

4.  **Page Number Identification & Formatting:**
    *   Carefully examine the image, especially the header or footer, for a page number.
    *   **If a page number is clearly visible:** Extract it. Include any surrounding static text if it's clearly part of the page number block (e.g., 'Page 5', '7 | 2023-2024 Handbook', 'Handbook 2024 | 12'). Format this entire identifiable block as a single string for the `page_number` field.
    *   **If no page number is identifiable:** Use the exact string `"-1"` for the `page_number` field.

5.  **Markdown Conversion (Minimal & Structural):**
    *   Convert the extracted `text_segment` to basic Markdown for structure and readability.
    *   Use `#`, `##`, `###`, etc., for headings and subheadings present within the chunk.
    *   Use standard Markdown for lists (`*` or `-` for unordered, `1.`, `2.` for ordered).
    *   Separate distinct paragraphs within the chunk with a single blank line.
    *   Preserve bolding (`**text**`) or italics (`*text*`) if they emphasize key terms or rules (e.g., `**must**`, `*required*`).
    *   **Do NOT attempt complex Markdown like tables.** If a table is present, extract its content row by row or cell by cell as plain text lines, preserving the logical association as best as possible. If extraction is too complex, briefly mention the table's topic in the `context` and extract only the caption or surrounding text if feasible.

6.  **Incomplete Content Handling:**
    *   If a chunk clearly represents content that is cut off mid-sentence or mid-thought *at the bottom of the page* (suggesting it continues on the next page), append the tag `[content_incomplete]` to the *very end* of the `text_segment`.
    *   Do *not* add this tag for cover pages, tables of contents, or intentionally short sections. Use it only when text seems unintentionally truncated due to the page break.

7.  **Output Format:** Structure your output strictly as a JSON object containing a list of chunks found on the page. Each chunk object must have the following keys:

    ```json
    {
      "segments": [
        {
          "text_segment": "[Extracted text segment with minimal Markdown formatting. Potentially ends with [content_incomplete]]",
          "context": "[Concise contextual description, e.g., 'Section Heading - Topic']",
          "page_number": "[Extracted page number string, or '-1']"
        },
        {
          "text_segment": "[Another extracted text segment...]",
          "context": "[Context for the second segment...]",
          "page_number": "[Extracted page number string, or '-1']"
        }
        // ... more chunks from the same page if applicable
      ]
    }
    ```

**Process the provided image page now according to these instructions.**
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
        response = provider.vision(image_path, ocr_prompt, response_format=PageSegmentsSchema, temperature=0.45)
        
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
