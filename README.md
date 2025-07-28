# PDF Outline Extractor

This project is a solution for the hackathon challenge to extract a structured outline (Title, H1, H2, H3) from PDF documents.

## Our Approach: A Hybrid System

This solution uses a sophisticated **hybrid approach** that combines the strengths of a deep learning model with an intelligent rule-based parser to achieve high accuracy and robustness.

The process for each PDF is as follows:

1.  **First Pass (AI Model):** A custom, layout-aware transformer model (`MiniLayoutLM` based on `google/electra-small-discriminator`) performs an initial analysis of the document. Having been trained on the DocLayNet dataset, this model understands the semantic context of headings and can identify them based on both their text and visual properties (font size, position, etc.).

2.  **Second Pass (Intelligent Rules):** A rule-based parser then analyzes the document's structure. It uses statistical analysis of font sizes to reliably find the document's main title and a scoring system to identify explicitly structured headings (e.g., numbered sections). It is also designed to detect and ignore common noise like Tables of Contents, headers, and footers.

3.  **Intelligent Merging:** The results from both passes are intelligently merged. The rule-based parser's high-confidence findings (like the title and numbered headings) are prioritized, and the AI model's results are used to add semantically identified headings that the rules might have missed. The final list is de-duplicated and sorted to produce a clean, accurate outline.

This hybrid system allows the solution to handle a wide variety of document layouts, from highly structured reports to more loosely formatted articles.

## Models and Libraries Used

*   **Core Model:** A custom `MiniLayoutLM` based on `google/electra-small-discriminator`, fine-tuned on the DocLayNet dataset. The final quantized model is **~14 MB**, well under the 200 MB limit.
*   **Key Libraries:**
    *   `PyTorch`: For building and running the deep learning model.
    *   `transformers`: For leveraging the pre-trained ELECTRA model.
    *   `PyMuPDF` (`fitz`): For fast and efficient PDF parsing.
    *   `NumPy`: For statistical analysis in the rule-based parser.

## How to Build and Run the Solution

### Prerequisites

*   Docker Desktop installed and running.

### 1. Build the Docker Image

Navigate to the root directory of this project in your terminal and run the following command. This will build the Docker image and install all dependencies.

```bash
docker build --platform linux/amd64 -t my-solution .