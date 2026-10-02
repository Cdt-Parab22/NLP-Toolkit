"""
NLP Toolkit - single-file Streamlit project

Install: pip install streamlit scikit-learn vaderSentiment numpy pandas youtube-transcript-api
Run:     streamlit run nlp_toolkit.py
"""
from __future__ import annotations

import re
from collections import Counter
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None


st.set_page_config(page_title="NLP Toolkit", page_icon="🧠", layout="wide")

SAMPLE = (
    "Natural language processing is a field of artificial intelligence that helps computers "
    "understand human language. It powers search engines, chatbots, translation systems and "
    "voice assistants. Modern NLP relies on machine learning, especially deep learning models "
    "such as transformers. These models learn patterns from huge amounts of text. However, "
    "they can also inherit bias from their training data. Researchers are working on making "
    "NLP systems fairer, more efficient and easier to explain. Businesses use NLP to analyse "
    "customer reviews, automate support and extract insights from documents."
)

# A practical limit for ordinary article and product pages.  Keeping a ceiling prevents
# an accidental scrape of a very large download from freezing the local app.
MAX_SCRAPE_BYTES = 15_000_000


# ---------------------------- NLP helpers ----------------------------
@st.cache_resource
def get_vader() -> SentimentIntensityAnalyzer:
    """Create the analyser once per Streamlit server."""
    return SentimentIntensityAnalyzer()


def split_sentences(text: str) -> list[str]:
    """Return useful sentences while handling newlines and short fragments."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if len(tokenize(part)) >= 3]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentiment(text: str) -> tuple[str, float]:
    score = get_vader().polarity_scores(text)["compound"]
    label = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
    return label, score


def summarize(text: str, sentence_count: int = 3) -> list[str]:
    """Extractive TextRank-style summary using TF-IDF sentence similarity."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= sentence_count:
        return sentences

    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:  # Empty vocabulary (for example, input contains only stop words).
        return sentences[:sentence_count]

    similarities = cosine_similarity(matrix)
    np.fill_diagonal(similarities, 0)
    row_sums = similarities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = similarities / row_sums

    number_of_sentences = len(sentences)
    damping = 0.85
    ranks = np.ones(number_of_sentences) / number_of_sentences
    for _ in range(50):
        ranks = (1 - damping) / number_of_sentences + damping * (transition.T @ ranks)

    selected_indexes = sorted(np.argsort(-ranks)[:sentence_count])
    return [sentences[index] for index in selected_indexes]


def extract_keywords(text: str, limit: int = 10) -> pd.DataFrame:
    sentences = split_sentences(text) or [text]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return pd.DataFrame(columns=["keyword", "score"])

    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    indexes = scores.argsort()[::-1][:limit]
    return pd.DataFrame({"keyword": terms[indexes], "score": scores[indexes].round(3)})


def count_syllables(word: str) -> int:
    count = len(re.findall(r"[aeiouy]+", word.lower()))
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability(text: str) -> tuple[float, str] | None:
    words = tokenize(text)
    if not words:
        return None

    sentence_count = max(len(re.findall(r"[.!?]+", text)), 1)
    syllables = sum(count_syllables(word) for word in words)
    score = 206.835 - 1.015 * (len(words) / sentence_count) - 84.6 * (syllables / len(words))

    if score >= 80:
        level = "Easy (6th grade)"
    elif score >= 60:
        level = "Standard (8th-9th grade)"
    elif score >= 30:
        level = "Difficult (college)"
    else:
        level = "Very difficult (graduate)"
    return round(score, 1), level


def similarity(first_text: str, second_text: str) -> tuple[float, list[str]]:
    """Return cosine similarity without crashing for blank or stop-word-only text."""
    if not first_text.strip() or not second_text.strip():
        return 0.0, []

    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([first_text, second_text])
    except ValueError:
        return 0.0, []

    score = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    shared_terms = sorted(
        (set(tokenize(first_text)) & set(tokenize(second_text))) - set(ENGLISH_STOP_WORDS)
    )
    return score, shared_terms


def top_words(text: str, limit: int = 10) -> list[tuple[str, int]]:
    words = tokenize(text)
    return Counter(
        word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2
    ).most_common(limit)


class PageContentParser(HTMLParser):
    """Small dependency-free parser for readable text, headings, and links."""

    ignored_tags = {"script", "style", "noscript", "svg", "template"}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.description = ""
        self.text_parts: list[str] = []
        self.headings: list[str] = []
        self.links: list[str] = []
        self._ignored_depth = 0
        self._in_title = False
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        if tag in self.ignored_tags:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = (attributes.get("name") or attributes.get("property") or "").lower()
            if name in {"description", "og:description"} and not self.description:
                self.description = (attributes.get("content") or "").strip()
        elif tag in {"h1", "h2", "h3"}:
            self._heading_tag = tag
            self._heading_parts = []
        elif tag == "a":
            href = (attributes.get("href") or "").strip()
            if href:
                absolute_url = urljoin(self.base_url, href)
                if urlparse(absolute_url).scheme in {"http", "https"}:
                    self.links.append(absolute_url)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag == self._heading_tag:
            heading = " ".join(self._heading_parts).strip()
            if heading:
                self.headings.append(heading)
            self._heading_tag = None
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        clean_data = " ".join(data.split())
        if not clean_data:
            return
        self.text_parts.append(clean_data)
        if self._in_title:
            self.title_parts.append(clean_data)
        if self._heading_tag:
            self._heading_parts.append(clean_data)


def clean_unique(values: list[str]) -> list[str]:
    """Remove blanks and preserve the original order of unique values."""
    return list(dict.fromkeys(value for value in values if value))


def scrape_page(url: str) -> dict[str, object]:
    """Fetch one public HTML page and extract its readable content."""
    requested_url = url.strip()
    if not requested_url:
        raise ValueError("Enter a website address first.")
    if not urlparse(requested_url).scheme:
        requested_url = f"https://{requested_url}"

    parsed_url = urlparse(requested_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Use a complete http:// or https:// website address.")

    request = Request(
        requested_url,
        headers={"User-Agent": "NLP-Toolkit-Educational-Scraper/1.0"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"This URL returned {content_type}, not an HTML page.")
            content = response.read(MAX_SCRAPE_BYTES + 1)
            if len(content) > MAX_SCRAPE_BYTES:
                raise ValueError("The page is larger than the 15 MB scraping limit.")
            encoding = response.headers.get_content_charset() or "utf-8"
            html = content.decode(encoding, errors="replace")
            final_url = response.geturl()
    except HTTPError as error:
        raise ValueError(f"The website returned HTTP {error.code} ({error.reason}).") from error
    except URLError as error:
        raise ValueError(f"Could not reach that website: {error.reason}") from error
    except TimeoutError as error:
        raise ValueError("The website took too long to respond.") from error

    parser = PageContentParser(final_url)
    parser.feed(html)
    parser.close()

    headings = clean_unique(parser.headings)
    text = " ".join(parser.text_parts)
    title = " ".join(parser.title_parts).strip() or (headings[0] if headings else "Untitled page")
    return {
        "url": final_url,
        "title": title,
        "description": parser.description,
        "text": text,
        "headings": headings,
        "links": clean_unique(parser.links),
    }


def extract_youtube_video_id(value: str) -> str:
    """Accept a normal YouTube URL, shortened URL, or a bare 11-character ID."""
    candidate = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate
    if not candidate:
        raise ValueError("Paste a YouTube video link or video ID.")

    if not urlparse(candidate).scheme:
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    path_parts = [part for part in parsed.path.split("/") if part]
    video_id = ""

    if host == "youtu.be" and path_parts:
        video_id = path_parts[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            video_id = path_parts[1]

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("That does not look like a valid YouTube video link or ID.")
    return video_id


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_youtube_transcript(video_id: str, language_codes: tuple[str, ...]) -> dict[str, object]:
    """Fetch available public captions and cache them for one hour."""
    if YouTubeTranscriptApi is None:
        raise ValueError(
            "The transcript feature needs one extra package. Run: pip install youtube-transcript-api"
        )

    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            fetched = api.fetch(video_id, languages=list(language_codes))
            raw_items = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
            language = getattr(fetched, "language", "Unknown")
            language_code = getattr(fetched, "language_code", "")
            is_generated = getattr(fetched, "is_generated", None)
        else:  # Support older installed releases of youtube-transcript-api.
            raw_items = YouTubeTranscriptApi.get_transcript(video_id, languages=list(language_codes))
            language = language_codes[0] if language_codes else "Unknown"
            language_code = language_codes[0] if language_codes else ""
            is_generated = None
    except Exception as error:
        raise ValueError(
            "No transcript could be retrieved. The video may have no public captions, be restricted, "
            f"or YouTube may have blocked the request. Details: {error}"
        ) from error

    snippets = []
    for item in raw_items:
        if isinstance(item, dict):
            text = item.get("text", "")
            start = item.get("start", 0)
            duration = item.get("duration", 0)
        else:
            text = getattr(item, "text", "")
            start = getattr(item, "start", 0)
            duration = getattr(item, "duration", 0)
        clean_text = " ".join(str(text).split())
        if clean_text:
            snippets.append(
                {"text": clean_text, "start": float(start or 0), "duration": float(duration or 0)}
            )

    if not snippets:
        raise ValueError("The video returned an empty transcript.")
    return {
        "video_id": video_id,
        "language": language,
        "language_code": language_code,
        "is_generated": is_generated,
        "snippets": snippets,
    }


def summarize_long_text(text: str, sentence_count: int) -> list[str]:
    """Summarize long transcripts in sections to avoid a huge similarity matrix."""
    sentences = split_sentences(text)
    if len(sentences) <= 180:
        return summarize(text, sentence_count)

    chunk_size = 150
    chunks = [sentences[index : index + chunk_size] for index in range(0, len(sentences), chunk_size)]
    candidates: list[str] = []
    sentences_per_chunk = max(2, min(8, int(np.ceil(sentence_count / len(chunks))) + 1))
    for chunk in chunks:
        candidates.extend(summarize(" ".join(chunk), sentences_per_chunk))
    return summarize(" ".join(candidates), sentence_count)


def create_ai_handoff(
    video_url: str,
    video_id: str,
    language: str,
    summary: str,
    timestamped_transcript: str,
) -> str:
    """Create a portable prompt that can be pasted into any AI assistant."""
    return (
        "# YouTube video context\n\n"
        f"Source: {video_url}\n"
        f"Video ID: {video_id}\n"
        f"Caption language: {language}\n\n"
        "## Summary\n"
        f"{summary or 'No summary was generated.'}\n\n"
        "## Full transcript with timestamps\n"
        f"{timestamped_transcript}\n\n"
        "## Task for the AI\n"
        "Use the video context above as the primary source. Clearly state any uncertainty caused by "
        "caption errors or missing context. Then help me with this task: [write your request here]"
    )


# ---------------------------- UI ----------------------------
st.title("🧠 NLP Toolkit")
st.caption("Sentiment · Summarizer · Keywords · Similarity · Text Stats | runs locally")

tabs = st.tabs(
    [
        "😊 Sentiment",
        "📝 Summarizer",
        "🔑 Keywords",
        "🔍 Similarity",
        "📊 Text Stats",
        "🌐 Web Scraper",
        "▶️ YouTube Transcript",
    ]
)


with tabs[0]:
    st.subheader("Sentiment analysis")
    mode = st.radio(
        "Input", ["Type or paste (one review per line)", "Upload CSV"], horizontal=True, key="sentiment_mode"
    )
    texts: list[str] = []

    if mode.startswith("Type"):
        raw_text = st.text_area(
            "Reviews, tweets, or feedback",
            "The battery life is amazing and the screen is gorgeous!\n"
            "Delivery was late and the packaging was damaged.\n"
            "It's okay, does the job.",
            height=150,
            key="sentiment_text",
        )
        texts = [line.strip() for line in raw_text.splitlines() if line.strip()]
    else:
        uploaded_csv = st.file_uploader("CSV file", type="csv", key="sentiment_csv")
        if uploaded_csv is not None:
            try:
                input_frame = pd.read_csv(uploaded_csv)
                if input_frame.empty or len(input_frame.columns) == 0:
                    st.warning("The CSV has no rows or columns.")
                else:
                    text_column = st.selectbox("Text column", input_frame.columns, key="sentiment_column")
                    texts = input_frame[text_column].dropna().astype(str).tolist()
            except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError) as error:
                st.error(f"The CSV could not be read: {error}")

    if st.button("Analyse sentiment", type="primary", key="analyse_sentiment"):
        if not texts:
            st.warning("Add at least one non-empty piece of text before analysing it.")
        else:
            rows = [(text, *sentiment(text)) for text in texts]
            result_frame = pd.DataFrame(rows, columns=["text", "sentiment", "score"])
            left, right = st.columns([1, 2])
            with left:
                st.bar_chart(result_frame["sentiment"].value_counts())
                st.metric("Average score", f"{result_frame['score'].mean():.3f}")
            with right:
                st.dataframe(result_frame, use_container_width=True, hide_index=True)
            st.download_button(
                "Download results (CSV)",
                result_frame.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )


with tabs[1]:
    st.subheader("Extractive summarizer (TextRank)")
    summary_text = st.text_area("Paste an article", SAMPLE, height=200, key="summary_text")
    sentence_total = st.slider("Sentences in summary", 1, 8, 3, key="summary_count")
    if st.button("Summarize", type="primary", key="summarize"):
        summary = summarize(summary_text, sentence_total)
        if not summary:
            st.warning("Please enter text with at least one sentence of three or more words.")
        else:
            result = " ".join(summary)
            st.success(result)
            st.caption(f"Reduced {len(tokenize(summary_text))} words to {len(tokenize(result))} words.")


with tabs[2]:
    st.subheader("Keyword and keyphrase extraction (TF-IDF)")
    keyword_text = st.text_area("Paste text", SAMPLE, height=200, key="keyword_text")
    keyword_total = st.slider("Number of keywords", 3, 25, 10, key="keyword_count")
    if st.button("Extract keywords", type="primary", key="extract_keywords"):
        keyword_frame = extract_keywords(keyword_text, keyword_total)
        if keyword_frame.empty:
            st.warning("Enter text containing words other than common stop words.")
        else:
            left, right = st.columns(2)
            left.dataframe(keyword_frame, use_container_width=True, hide_index=True)
            right.bar_chart(keyword_frame.set_index("keyword"))


with tabs[3]:
    st.subheader("Text similarity / plagiarism check")
    left, right = st.columns(2)
    with left:
        text_a = st.text_area("Text A", SAMPLE, height=180, key="similarity_a")
    with right:
        text_b = st.text_area(
            "Text B",
            "NLP lets machines understand language and is used in chatbots, "
            "translation and search. Deep learning and transformers drive modern systems.",
            height=180,
            key="similarity_b",
        )
    if st.button("Compare", type="primary", key="compare"):
        score, shared_terms = similarity(text_a, text_b)
        st.metric("Cosine similarity", f"{score * 100:.1f}%")
        st.progress(int(round(score * 100)))
        verdict = "Very similar" if score > 0.70 else "Somewhat related" if score > 0.35 else "Mostly different"
        st.write(f"**Verdict:** {verdict}")
        st.write("**Shared terms:**", ", ".join(shared_terms) or "none")


with tabs[4]:
    st.subheader("Readability and text statistics")
    statistics_text = st.text_area("Paste text", SAMPLE, height=200, key="statistics_text")
    if st.button("Analyse text", type="primary", key="analyse_text"):
        words = tokenize(statistics_text)
        result = readability(statistics_text)
        if result is None:
            st.warning("Please enter some text.")
        else:
            flesch_score, reading_level = result
            metrics = st.columns(4)
            metrics[0].metric("Words", len(words))
            metrics[1].metric("Sentences", max(len(split_sentences(statistics_text)), 1))
            metrics[2].metric("Unique words", len(set(words)))
            metrics[3].metric("Flesch score", flesch_score, reading_level, delta_color="off")

            frequent_words = top_words(statistics_text)
            st.write("**Top 10 words (stop words removed)**")
            if frequent_words:
                word_frame = pd.DataFrame(frequent_words, columns=["word", "count"]).set_index("word")
                st.bar_chart(word_frame)
            else:
                st.info("No non-stop words were found to chart.")


with tabs[5]:
    st.subheader("Web scraper")
    st.caption("Extract text from one public, static HTML page at a time. Check the website's rules before scraping.")
    page_url = st.text_input(
        "Website address",
        placeholder="https://example.com/article",
        key="scraper_url",
    )
    if st.button("Scrape page", type="primary", key="scrape_page"):
        with st.spinner("Fetching and extracting page content..."):
            try:
                page = scrape_page(page_url)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success("Page scraped successfully.")
                st.markdown(f"### {page['title']}")
                st.caption(str(page["url"]))
                if page["description"]:
                    st.write(page["description"])

                text = str(page["text"])
                headings = list(page["headings"])
                links = list(page["links"])
                metrics = st.columns(3)
                metrics[0].metric("Words extracted", len(tokenize(text)))
                metrics[1].metric("Headings", len(headings))
                metrics[2].metric("Links", len(links))

                st.text_area("Extracted text", text, height=260, key="scraped_text")
                if headings:
                    with st.expander("Page headings"):
                        st.dataframe(pd.DataFrame({"heading": headings}), hide_index=True, use_container_width=True)
                if links:
                    with st.expander("Links found"):
                        st.dataframe(pd.DataFrame({"url": links}), hide_index=True, use_container_width=True)

                download_text = f"Title: {page['title']}\nURL: {page['url']}\n\n{text}"
                st.download_button(
                    "Download extracted text",
                    download_text.encode("utf-8"),
                    file_name="scraped_page.txt",
                    mime="text/plain",
                    key="download_scraped_text",
                )
                st.download_button(
                    "Download links (CSV)",
                    pd.DataFrame({"url": links}).to_csv(index=False).encode("utf-8"),
                    file_name="scraped_links.csv",
                    mime="text/csv",
                    key="download_scraped_links",
                )


with tabs[6]:
    st.subheader("YouTube transcript and AI handoff")
    st.caption(
        "Fetches public captions only. Use content you are allowed to reuse, and review captions for errors."
    )
    youtube_url = st.text_input(
        "YouTube video link or ID",
        placeholder="https://www.youtube.com/watch?v=...",
        key="youtube_url",
    )
    language_input = st.text_input(
        "Caption language preference",
        value="en",
        help="Enter language codes in order of preference, separated by commas. Example: en, hi",
        key="youtube_languages",
    )
    summary_count = st.slider(
        "Summary sentences", min_value=3, max_value=20, value=8, key="youtube_summary_count"
    )

    if st.button("Get transcript and summarize", type="primary", key="get_youtube_transcript"):
        try:
            video_id = extract_youtube_video_id(youtube_url)
            languages = tuple(
                code.strip().lower() for code in language_input.split(",") if code.strip()
            ) or ("en",)
            with st.spinner("Retrieving public captions and building the summary..."):
                transcript_data = fetch_youtube_transcript(video_id, languages)
        except ValueError as error:
            st.error(str(error))
        else:
            snippets = list(transcript_data["snippets"])
            timestamped_transcript = "\n".join(
                f"[{format_timestamp(float(snippet['start']))}] {snippet['text']}" for snippet in snippets
            )
            plain_transcript = " ".join(str(snippet["text"]) for snippet in snippets)
            summary_sentences = summarize_long_text(plain_transcript, summary_count)
            summary = " ".join(summary_sentences)
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            ai_handoff = create_ai_handoff(
                video_url,
                video_id,
                str(transcript_data["language"]),
                summary,
                timestamped_transcript,
            )

            st.success("Transcript and AI-ready handoff are ready.")
            generated_note = transcript_data["is_generated"]
            caption_type = "Auto-generated" if generated_note is True else "Manual or unspecified"
            metrics = st.columns(3)
            metrics[0].metric("Caption segments", len(snippets))
            metrics[1].metric("Transcript words", len(tokenize(plain_transcript)))
            metrics[2].metric("Caption type", caption_type)
            st.markdown("### Summary")
            st.write(summary)

            st.markdown("### AI-ready handoff")
            st.caption("Use the copy icon below, or select the text and paste it into another AI agent.")
            st.code(ai_handoff, language=None)
            st.download_button(
                "Download AI handoff (.txt)",
                ai_handoff.encode("utf-8"),
                file_name=f"youtube_ai_handoff_{video_id}.txt",
                mime="text/plain",
                key="download_youtube_handoff",
            )

            with st.expander("Full timestamped transcript"):
                st.code(timestamped_transcript, language=None)
            st.download_button(
                "Download full transcript (.txt)",
                timestamped_transcript.encode("utf-8"),
                file_name=f"youtube_transcript_{video_id}.txt",
                mime="text/plain",
                key="download_youtube_transcript",
            )
