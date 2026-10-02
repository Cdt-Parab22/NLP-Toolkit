# 🧠 NLP Toolkit

An all-in-one Streamlit application for analysing, summarising, and extracting insights from text.

## Features

- 😊 **Sentiment Analysis**  
  Analyse individual reviews or a batch of text from a CSV file.

- 📝 **Text Summarizer**  
  Create extractive summaries using a TextRank-style TF-IDF approach.

- 🔑 **Keyword Extractor**  
  Find important keywords and keyphrases from any text.

- 🔍 **Text Similarity Checker**  
  Compare two texts with cosine similarity for plagiarism or content-overlap checks.

- 📊 **Readability & Text Statistics**  
  View word count, sentence count, unique words, frequent terms, and Flesch readability score.

- 🌐 **Web Scraper**  
  Extract visible text, headings, and links from public HTML webpages.

- ▶️ **YouTube Transcript & AI Handoff**  
  Fetch public YouTube captions, generate a summary, and create an AI-ready context block that can be copied or downloaded for use in ChatGPT or another AI assistant.

## Built With

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- VADER Sentiment
- YouTube Transcript API

## Installation

Clone the repository:

```bash
https://github.com/Cdt-Parab22/NLP-Toolkit.git
cd YOUR-REPOSITORY
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
streamlit run nlp_toolkit.py
```

Then open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

## Usage

1. Open a tool from the navigation tabs.
2. Paste text, upload a CSV, enter a webpage URL, or add a YouTube video link.
3. Run the analysis.
4. Download results or copy the AI-ready output when available.

## Notes

- The web scraper works with public static HTML pages.
- The YouTube tool uses publicly available captions. Videos without captions, restricted videos, or blocked requests may not return a transcript.
- Always respect website terms, copyright rules, and content ownership when scraping or reusing content.

## License

This project is available for educational and personal use. Add an MIT License if you want others to freely reuse and improve the project.
