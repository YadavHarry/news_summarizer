from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
import ssl
import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import os

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
print("Initializing NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def fetch_article_bs4(url):
    """Fetch article content from URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.text.strip() if title_tag else "No title found"

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try to extract article content from common article containers
        article_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '.story-body',
            '.article-body'
        ]

        article_text = ""
        for selector in article_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                paragraphs = content_div.find_all(['p'])
                if paragraphs:
                    article_text = " ".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                    break

        # Fallback: extract all paragraphs
        if not article_text:
            article_tags = soup.find_all(['p'])
            article_text = " ".join(p.get_text().strip() for p in article_tags if p.get_text().strip())

        # Clean up the text
        article_text = ' '.join(article_text.split())

        return title, article_text

    except requests.exceptions.RequestException as e:
        return None, f"Network Error: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def summarize_text(text, sentence_count=5):
    """Summarize text using Sumy"""
    try:
        # Check if text is long enough
        if len(text.split()) < 20:
            return "Text too short to summarize effectively.", False

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()

        # Ensure we don't ask for more sentences than available
        sentences_in_text = len(parser.document.sentences)
        actual_sentence_count = min(sentence_count, sentences_in_text)

        if actual_sentence_count == 0:
            return "No valid sentences found in the text.", False

        summary = summarizer(parser.document, actual_sentence_count)
        summary_text = " ".join(str(sentence) for sentence in summary)

        return summary_text, True

    except Exception as e:
        return f"Summarization error: {str(e)}", False


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/summarize-url', methods=['POST'])
def summarize_url_endpoint():
    """API endpoint to summarize article from URL"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        sentence_count = int(data.get('sentences', 5))

        if not url:
            return jsonify({
                'success': False,
                'error': 'URL is required'
            }), 400

        # Add https:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Fetch article
        title, content = fetch_article_bs4(url)

        if not content or "Error" in content:
            return jsonify({
                'success': False,
                'error': content or 'Could not fetch article content'
            }), 400

        if len(content.strip()) < 100:
            return jsonify({
                'success': False,
                'error': 'Very little content extracted. The website might be blocking scraping or using dynamic content.'
            }), 400

        # Summarize
        summary, success = summarize_text(content, sentence_count)

        if not success:
            return jsonify({
                'success': False,
                'error': summary
            }), 400

        # Calculate stats
        stats = {
            'originalLength': len(content),
            'wordCount': len(content.split()),
            'sentences': len([s for s in content.split('.') if s.strip()])
        }

        return jsonify({
            'success': True,
            'title': title,
            'summary': summary,
            'stats': stats
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/summarize-text', methods=['POST'])
def summarize_text_endpoint():
    """API endpoint to summarize provided text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        sentence_count = int(data.get('sentences', 5))

        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400

        if len(text) < 100:
            return jsonify({
                'success': False,
                'error': 'Text is too short. Please enter at least 100 characters.'
            }), 400

        # Summarize
        summary, success = summarize_text(text, sentence_count)

        if not success:
            return jsonify({
                'success': False,
                'error': summary
            }), 400

        # Calculate stats
        stats = {
            'originalLength': len(text),
            'wordCount': len(text.split()),
            'sentences': len([s for s in text.split('.') if s.strip()])
        }

        return jsonify({
            'success': True,
            'title': 'Custom Text Analysis',
            'summary': summary,
            'stats': stats
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'News Summarizer API is running'
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("🚀 Starting News Summarizer API...")
    print("📡 Server will be available at: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   - POST /api/summarize-url")
    print("   - POST /api/summarize-text")
    print("   - GET /api/health")

    app.run(debug=True, host='0.0.0.0', port=5000)