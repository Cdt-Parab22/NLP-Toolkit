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

# Proprietary Use and Attribution License

Copyright (c) 2026 Pranav Parab. All rights reserved.

## 1. Permission to use

Permission is granted to download, install, and use an unmodified copy of this software solely for personal, educational, or internal use.

This permission does not transfer ownership of the software or its source code.

## 2. Restrictions

Without prior written permission from the copyright owner, you may not:

- Copy, reproduce, modify, adapt, translate, or create derivative works from the source code.
- Redistribute, publish, sublicense, sell, rent, lease, or commercially exploit the software or any substantial part of its source code.
- Remove, alter, or hide copyright, authorship, or attribution notices.
- Present the software, its source code, or a modified version as your own work.
- Claim authorship, ownership, or original credit for this project.
- Use the project name, branding, or author name in a way that suggests endorsement without written permission.

## 3. Attribution

Any permitted public reference to this software must acknowledge:

> Created by Pranav Parab.

## 4. Ownership

All intellectual-property rights, including copyright, source code, branding, design, and documentation, remain the exclusive property of Pranav Parab.

## 5. Termination

This permission ends automatically if any term of this license is violated. Upon termination, you must stop using the software and delete all copies in your possession or control.

## 6. Disclaimer

This software is provided “as is,” without warranty of any kind. The copyright owner is not liable for any claim, damages, or other liability arising from the use of this software.
