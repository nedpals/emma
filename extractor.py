from typing import List, Generator, Iterable, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from bs4 import BeautifulSoup, element
import os
import re

class PDFToMarkdownExtract:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = PDFMinerPDFasHTMLLoader(self.file_path)
        self.result = self.loader.load()[0]
        self.soup = BeautifulSoup(self.result.page_content, 'html.parser')
        self.raw_pages = self.soup.find("body").find_all("span", recursive=False)
        self.pages = []

    def extract_content(self) -> List[Document]:
        # each span represents a page
        page_nr = 0

        for page in self.raw_pages:
            # Smart cast to Tag
            if type(page) != element.Tag:
                continue

            # Do not iterate on spans that have content
            elif len(page.contents) != 0:
                continue

            # Skip other spans that do not have the left:0px; style
            elif 'left:0px;' not in page.attrs.get("style", ""):
                continue

            # Increment the page number
            page_nr += 1

            # Skip the table of contents from page 2 to 5
            if page_nr > 1 and page_nr < 6:
                continue

            # Get the content of the page
            children = []
            sib = page.next_sibling

            while sib != None:
                if type(sib) == element.Tag and sib.name == "span" and len(sib.contents) == 0:
                    break
                elif type(sib) == element.NavigableString and sib == "\n":
                    sib = sib.next_sibling
                    continue
                elif sib.attrs.get("style", "") == "position:absolute; top:0px;":
                    break

                children.append(sib)
                sib = sib.next_sibling

            if len(children) == 0:
                continue

            # Add + 2 to page_nr to account for the covers
            for processed in self.process(page_nr + 2, children, self.result.metadata):
                self.pages.append(processed)
                break

        return self.pages

    def parse_inline_styles(style: str) -> dict:
        styles = style.split(";")
        entries = {}
        for style in styles:
            if style.strip() == "":
                continue
            key, val = style.split(":")
            entries[key.strip()] = val.strip()
        return entries

    def get_text(tag: element.Tag):
        children = tag.contents
        content = ""
        for child in children:
            if type(child) == element.NavigableString:
                content += child
            elif type(child) == element.Tag and child.name == "br":
                content += "\n"
            else:
                break
        return content

    def extract_styles_and_font_info(self, tag: element.Tag):
        raw_styles = tag.attrs.get("style", "")
        styles = self.parse_inline_styles(raw_styles)

        # Font info
        font = str(styles.get("font-family", ""))
        text_size = styles.get("font-size", "11px")

        # Parse text_size to integer
        text_size = re.sub(r'[^0-9]', '', text_size)
        text_size_px = int(text_size) # in pixels
        return styles, font, text_size_px

    def style_with_markdown(self, tag: element.Tag):
        # Skip line breaks
        if tag.name == "br":
            return "", ""

        # for reference only.
        ref_text = self.get_text(tag)
        _, font, text_size_px = self.extract_styles_and_font_info(tag)

        left = ""
        right = ""

        # For markdown generation
        heading_level = 0
        is_bold = False
        is_italic = False

        if len(ref_text) != 0 and ref_text[0] == "•":
            return "- ", "\n"

        # Detection
        if text_size_px == 12:
            if font == 'MyriadPro-Bold':
                heading_level = 1
            elif font == 'MyriadPro-Regular' and ref_text.isupper():
                heading_level = 2
            elif font == 'MinionPro-Regular':
                heading_level = 3
        elif text_size_px == 11:
            if font == 'MyriadPro-Bold':
                if ref_text.isupper():
                    heading_level = 3
                else:
                    heading_level = 4

        if font == 'MinionPro-Bold' or (font == 'MyriadPro-Bold' and text_size_px < 11):
            is_bold = True

        if font == 'MinionPro-BoldIt' or font == 'MinionPro-SemiboldIt':
            is_bold = True
            is_italic = True

        # Generate text goes here
        if heading_level > 0:
            left = "#" * min(heading_level, 6) + " "

        if is_bold:
            left += "**"
            right += "**"

        if is_italic:
            left += "*"
            right += "*"

        if heading_level > 0:
            right += "\n"

        return left, right

    def to_markdown_multiple(self, children: Iterable[Any]) -> str:
        content = ""

        for child in children:
            if type(child) == element.NavigableString:
                if child.text[0] == "•" or child.text == "\n":
                    continue

                text = child.text
                if text[-1] == "\n":
                    text = text[:-1]

                if child.next_sibling and type(child.next_sibling) == element.Tag and child.next_sibling.name == "br":
                    next_sib = child.next_sibling
                    if next_sib.next_sibling and type(next_sib.next_sibling) == element.NavigableString and len(next_sib.next_sibling.text) != 0 and next_sib.next_sibling.text[0].isupper():
                        # Add newline if next sibling is a new sentence
                        text += "\n"

                content += text
            elif type(child) != element.Tag:
                continue
            elif child.name == "div" and child.attrs.get("style", "") == "position:absolute; top:0px;" and len(child.contents) != 0 and child.contents[0].text.startswith("Page: "):
                continue
            elif self.has_page_indicator(child):
                continue
            else:
                content += self.to_markdown(child)


            # add newline only if div or has punctuation
            if child.name == "div" or (child.parent.name == "div" and len(child.parent.contents) > 1) or (len(content) != 0 and content[-1] in [".", "!", "?"]):
                content += "\n"
            elif child.next_sibling and type(child) == element.Tag and child.next_sibling.name == child.name:
                content += "\n"

        return content

    def to_markdown(self, tag: element.Tag):
        _, font, _ = self.extract_styles_and_font_info(tag)
        if font == 'NexaBold' or font == 'NexaLight':
            # Footer detected. Stop here.
            return ""

        left, right = self.style_with_markdown(tag)
        text = self.to_markdown_multiple(tag.contents)
        return left + text + right

    def has_page_indicator(self, tag: element.Tag) -> bool:
        if bool(re.match(r"^Page \d+$", tag.text)):
            return True
        return False

    def process(self, page_nr: int, children: Iterable[Any], metadata: dict) -> Generator[Document, None, None]:
        doc_metadata = { "page": page_nr }

        # Add existing metadata to the document
        for key, val in metadata.items():
            doc_metadata[key] = val

        content = self.to_markdown_multiple(children)

        if len(content) != 0:
            yield Document(page_content=content, metadata=doc_metadata)

pdf_path = "Handbook 2018.pdf"

def extract_content():
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(pdf_path).load()

def extract_content2():
    from llmsherpa.readers import LayoutPDFReader
    llmsherpa_api_base_url = os.environ.get("LLMSHERPA_API_URL", "http://127.0.0.1:5010")
    llmsherpa_api_url = f"{llmsherpa_api_base_url}/api/parseDocument?renderFormat=all&applyOcr=yes&useNewIndentParser=yes"

    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    doc = pdf_reader.read_pdf(pdf_path)
    return map(
        lambda chunk: Document(page_content=chunk.to_context_text()),
        doc.chunks())
