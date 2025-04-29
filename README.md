# emma

![Emma - The Interactive Handbook for Ignatian Marians](/.github/banner.png)

## 🌟 About Emma

Emma is an AI-powered interactive handbook designed specifically for Ignatian Marians at UIC. She provides instant answers about academic policies, campus life, and student services - no handbook skimming required.

## 🚀 Features

- **Academic Policy Guidance** - Get clear explanations about attendance, grading, and course requirements
- **Campus Life Information** - Learn about events, facilities, and resources available on campus
- **Student Services Support** - Navigate administrative processes, support services, and more
- **Natural Language Interface** - Ask questions in everyday language, just like chatting with a friend

## 🛠️ Technology

Emma is built using:
- [Google Gemma 3](https://ai.google.dev/gemma) / [Google Gemini](https://ai.google.dev/gemini) - For natural language understanding and generation
- [LM Studio](https://lmstudio.ai/) - For local model deployment and management
- [Tailwind CSS](https://tailwindcss.com/) - For responsive and elegant UI design
- [Vite](https://vitejs.dev/) - For lightning-fast frontend development
- [ChromaDB](https://www.trychroma.com/) - For vector database and semantic search capabilities

## 🚀 Installation & Setup

### Prerequisites
- Node.js (v18 or higher)
- Python (v3.9 or higher)
- Git
- [LM Studio](https://lmstudio.ai/) with the following models downloaded and available:
  - `gemma-3-4b-it-qat`
  - `gemma-3-12b-it-qat` (optional, for vision/OCR during ingestion)
  - `text-embedding-nomic-embed-text-v1.5`

### Local Setup
1. Clone the repository
   ```bash
   git clone https://github.com/nedpals/emma.git
   cd emma
   ```

2. Install dependencies
   ```bash
   # Install frontend dependencies
   cd frontend
   npm install

   # Install backend dependencies
   cd ..
   pip install -r requirements.txt
   ```

3. Start the development servers
   ```bash
   # Start the frontend development server (in frontend directory)
   cd frontend
   npm run dev

   # In another terminal, start the backend server
   python main.py
   ```

4. Access Emma at `http://localhost:8000`

### Handbook Ingestion

Emma uses ChromaDB as its vector store to enable semantic search capabilities. There are two primary methods for ingesting handbook content:

**Method 1: Using LM Studio (Recommended for local processing)**

1.  Place your handbook documents (PDF format) in the project's root directory (e.g., `handbook.pdf`).
2.  Ensure LM Studio is running and serving the required models (`gemma-3-12b-it-qat` for vision and `text-embedding-nomic-embed-text-v1.5` for embeddings) at `http://localhost:1234`.
3.  Run the embedding script. Choose one of the following commands:
    *   **Standard Speed:** Processes documents in smaller batches (default: 2). Suitable for systems with limited resources.
        ```bash
        python embedding.py
        ```
    *   **Faster Speed:** Processes documents in larger batches (e.g., 600). Requires more system resources (RAM/VRAM) but significantly speeds up ingestion. Adjust the `MAX_EMBED_COUNT` value based on your system's capabilities.
        ```bash
        MAX_EMBED_COUNT=600 python embedding.py
        ```
4.  The script will first use the vision model (`gemma-3-12b-it-qat`) to extract text segments from each page of the PDF, caching the results in the `extracted_2` directory. Then, it will use the embedding model (`text-embedding-nomic-embed-text-v1.5`) to create vector embeddings for each segment.
5.  The embeddings and vector store data will be persisted in the `embeddings_db` directory.

**Method 2: Using Google AI Studio (Alternative for text extraction)**

This method is useful if you encounter issues with local vision model processing or prefer using Google's cloud-based models for the initial text extraction.

1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Create a new prompt. Upload your handbook PDF file.
3.  Use the prompt content from the `ingest_gemini_prompt.txt` file in this repository. Ensure you are using a capable multimodal model like **Gemini 2.5 Pro**.
4.  Run the prompt. Google AI Studio will process the PDF and generate a JSON output containing the extracted text segments based on the prompt's instructions.
5.  Copy the entire JSON output.
6.  Create a new file named `page_0.json` inside the `extracted_2` directory within your local project folder (create the `extracted_2` directory if it doesn't exist).
7.  Paste the copied JSON content into `extracted_2/page_0.json` and save the file.
8.  Ensure LM Studio is running and serving *only* the required embedding model (`text-embedding-nomic-embed-text-v1.5`) at `http://localhost:1234`.
9.  Run the embedding script (choose standard or faster speed as described in Method 1):
    ```bash
    # Standard speed
    python embedding.py
    # OR Faster speed
    # MAX_EMBED_COUNT=600 python embedding.py
    ```
10. The script will detect the cached data in `extracted_2/page_0.json`, skip the vision/OCR step, and proceed directly to embedding the text segments using the local embedding model.
11. The embeddings and vector store data will be persisted in the `embeddings_db` directory.

## 🤝 Contributing

We welcome contributions to make Emma even better! If you'd like to contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer
This project is not affiliated with, endorsed by, or connected to the University of the Immaculate Conception (UIC). Emma is an independent, personal project created with a strong desire to assist Ignatian Marians by utilizing the latest technologies available. All information provided should be verified with official UIC sources and personnel.

---

<p align="center">Made with ❤️ for Ignatian Marians</p>