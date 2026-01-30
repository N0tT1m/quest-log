import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import * as schema from '../src/lib/server/schema';

const sqlite = new Database('data/quest-log.db');

const db = drizzle(sqlite, { schema });

interface TaskData {
	title: string;
	description: string;
	details: string;
}

interface ModuleData {
	name: string;
	description: string;
	tasks: TaskData[];
}

interface PathData {
	name: string;
	description: string;
	language: string;
	color: string;
	skills: string;
	startHint: string;
	difficulty: string;
	estimatedWeeks: number;
	schedule: string;
	modules: ModuleData[];
}

const mlPipelinePath: PathData = {
	name: 'Complete ML Pipeline: Data to Deployment',
	description: 'Master the complete machine learning pipeline from raw data collection through cleaning, preprocessing, training, evaluation, to production deployment. Learn the reality: 80% of ML is data work.',
	language: 'Python',
	color: 'green',
	skills: 'data collection, data cleaning, preprocessing, tokenization, training pipelines, evaluation metrics, model deployment, MLOps, monitoring',
	startHint: 'Start by building a data collection pipeline with proper caching and rate limiting',
	difficulty: 'intermediate',
	estimatedWeeks: 10,
	schedule: `## 10-Week Complete ML Pipeline Mastery

### Weeks 1-2: Data Collection & Cleaning

#### Week 1: Data Collection
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Collection Setup | Build DataCollector base class |
| Tue | Web Scraping | Implement scrapers with rate limiting |
| Wed | APIs | Build API clients with caching |
| Thu | Datasets | Load from HuggingFace, Kaggle |
| Fri | Storage | Implement efficient data storage |
| Weekend | Pipeline | Build complete collection pipeline |

#### Week 2: Data Cleaning
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Text Cleaning | Remove HTML, URLs, normalize Unicode |
| Tue | Deduplication | Implement exact and fuzzy dedup |
| Wed | Quality Filters | Build content quality filters |
| Thu | Code Cleaning | Clean code-specific data |
| Fri | Validation | Add data validation checks |
| Weekend | Integration | Complete cleaning pipeline |

### Weeks 3-4: Preprocessing & Tokenization

#### Week 3: Data Preprocessing
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Normalization | Text normalization techniques |
| Tue | Feature Engineering | Extract relevant features |
| Wed | Augmentation | Implement data augmentation |
| Thu | Splitting | Train/val/test split strategies |
| Fri | Imbalance | Handle class imbalance |
| Weekend | Pipeline | Complete preprocessing pipeline |

#### Week 4: Tokenization
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | BPE | Implement Byte Pair Encoding |
| Wed | WordPiece | Build WordPiece tokenizer |
| Thu | SentencePiece | Implement unigram tokenizer |
| Fri | Training | Train custom tokenizer |
| Weekend | Integration | Add to data pipeline |

### Weeks 5-6: Training Pipeline

#### Week 5: Dataset & DataLoader
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Dataset Class | Build PyTorch Dataset |
| Tue | DataLoader | Implement efficient loading |
| Wed | Batching | Dynamic batching strategies |
| Thu | Collation | Custom collate functions |
| Fri | Streaming | Implement streaming datasets |
| Weekend | Optimization | Profile and optimize |

#### Week 6: Training Loop
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Basic Loop | Implement training loop |
| Tue | Mixed Precision | Add FP16/BF16 training |
| Wed | Gradient Accum | Handle large batches |
| Thu | Checkpointing | Save/load checkpoints |
| Fri | Logging | Add wandb/tensorboard |
| Weekend | Full Training | Train complete model |

### Weeks 7-8: Evaluation & Metrics

#### Week 7: Evaluation Pipeline
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Metrics | Implement common metrics |
| Tue | Classification | Precision, recall, F1, ROC-AUC |
| Wed | Generation | BLEU, ROUGE, perplexity |
| Thu | Regression | MSE, MAE, R² |
| Fri | Custom Metrics | Build domain-specific metrics |
| Weekend | Dashboard | Create evaluation dashboard |

#### Week 8: Model Analysis
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Error Analysis | Analyze failure cases |
| Tue | Bias Detection | Check for biases |
| Wed | Ablation Studies | Component importance |
| Thu | Interpretability | Add model explanations |
| Fri | Comparison | Compare multiple models |
| Weekend | Report | Complete evaluation report |

### Weeks 9-10: Deployment & MLOps

#### Week 9: Model Serving
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Export | TorchScript, ONNX export |
| Tue | Optimization | Quantization, pruning |
| Wed | API Server | Build FastAPI service |
| Thu | Batching | Implement request batching |
| Fri | Caching | Add result caching |
| Weekend | Testing | Load test deployment |

#### Week 10: MLOps & Monitoring
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Containerization | Docker, Kubernetes |
| Tue | Monitoring | Prometheus, Grafana setup |
| Wed | Logging | Structured logging |
| Thu | CI/CD | Automated deployment pipeline |
| Fri | Alerts | Set up alerting |
| Weekend | Production | Deploy to production |

### Daily Commitment: 2-3 hours
### Prerequisites: Python, PyTorch basics`,
	modules: [
		{
			name: 'Data Collection',
			description: 'Build robust data collection pipelines with proper rate limiting and caching',
			tasks: [
				{
					title: 'Build DataCollector base class with rate limiting and caching',
					description: 'Create reusable data collection infrastructure',
					details: `## Data Collection Infrastructure

### Why Data Collection Matters

**Reality Check:**
- 20% of time spent collecting data
- Most people think it's 5%
- Good collection saves hours of debugging later

### Base DataCollector Class

\`\`\`python
import requests
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Base class for data collection with rate limiting and caching.

    Features:
    - Automatic rate limiting
    - File-based caching
    - Error handling and retries
    - Progress tracking
    """

    def __init__(self, output_dir: str, rate_limit: float = 1.0, cache_enabled: bool = True):
        """
        Args:
            output_dir: Directory to save collected data
            rate_limit: Minimum seconds between requests
            cache_enabled: Whether to use caching
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit = rate_limit
        self.cache_enabled = cache_enabled
        self.last_request_time = 0

        # Stats
        self.stats = {
            'requests_made': 0,
            'cache_hits': 0,
            'errors': 0
        }

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time

        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path from URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.output_dir / 'cache' / f"{url_hash}.json"

    def _read_cache(self, url: str) -> Optional[Dict]:
        """Read from cache if exists."""
        if not self.cache_enabled:
            return None

        cache_path = self._get_cache_path(url)

        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for {url}")
                return json.load(f)

        return None

    def _write_cache(self, url: str, data: Dict):
        """Write to cache."""
        if not self.cache_enabled:
            return

        cache_path = self._get_cache_path(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def fetch(self, url: str, headers: Optional[Dict] = None, max_retries: int = 3) -> Dict:
        """
        Fetch URL with caching and rate limiting.

        Args:
            url: URL to fetch
            headers: Optional HTTP headers
            max_retries: Number of retry attempts

        Returns:
            Fetched data as dictionary
        """
        # Check cache first
        cached = self._read_cache(url)
        if cached is not None:
            return cached

        # Rate limiting
        self._rate_limit_wait()

        # Default headers
        if headers is None:
            headers = {
                'User-Agent': 'DataCollector/1.0 (Educational Purpose)'
            }

        # Retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching: {url} (attempt {attempt + 1}/{max_retries})")

                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()

                data = response.json()
                self.stats['requests_made'] += 1

                # Cache successful response
                self._write_cache(url, data)

                return data

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed: {e}")
                self.stats['errors'] += 1

                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts")
                    raise

    def save_data(self, data: Any, filename: str):
        """Save data to file."""
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(data))

        logger.info(f"Saved data to {filepath}")

    def print_stats(self):
        """Print collection statistics."""
        total = self.stats['requests_made'] + self.stats['cache_hits']
        cache_rate = (self.stats['cache_hits'] / total * 100) if total > 0 else 0

        print("\\n=== Collection Statistics ===")
        print(f"Total requests: {total}")
        print(f"API calls: {self.stats['requests_made']}")
        print(f"Cache hits: {self.stats['cache_hits']} ({cache_rate:.1f}%)")
        print(f"Errors: {self.stats['errors']}")


# Example: Wikipedia Data Collector
class WikipediaCollector(DataCollector):
    """Collect Wikipedia articles."""

    def __init__(self, output_dir: str = "data/wikipedia"):
        super().__init__(output_dir, rate_limit=0.1)  # Be nice to Wikipedia
        self.api_url = "https://en.wikipedia.org/w/api.php"

    def get_article(self, title: str) -> Dict:
        """Get a single article's content."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json"
        }

        url = f"{self.api_url}?" + "&".join(f"{k}={v}" for k, v in params.items())
        data = self.fetch(url)

        # Extract article text
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            return {
                'title': page.get('title'),
                'text': page.get('extract', ''),
                'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            }

        return {}

    def get_category_articles(self, category: str, limit: int = 100) -> list:
        """Get articles from a category."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": limit,
            "format": "json"
        }

        url = f"{self.api_url}?" + "&".join(f"{k}={v}" for k, v in params.items())
        data = self.fetch(url)

        members = data.get("query", {}).get("categorymembers", [])
        return [m['title'] for m in members]


# Example: HuggingFace Dataset Collector
class HuggingFaceCollector:
    """Collect datasets from HuggingFace."""

    def __init__(self, output_dir: str = "data/huggingface"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_name: str, split: str = "train", streaming: bool = False):
        """Download dataset from HuggingFace."""
        from datasets import load_dataset

        logger.info(f"Loading dataset: {dataset_name}")

        dataset = load_dataset(dataset_name, split=split, streaming=streaming)

        if not streaming:
            # Save to disk
            save_path = self.output_dir / dataset_name.replace('/', '_')
            dataset.save_to_disk(str(save_path))
            logger.info(f"Saved to {save_path}")

        return dataset


# Usage example
def example_collection():
    # Wikipedia collection
    wiki = WikipediaCollector()

    # Collect articles
    articles = []
    titles = ["Machine Learning", "Deep Learning", "Neural Network"]

    for title in titles:
        article = wiki.get_article(title)
        articles.append(article)

    # Save collected articles
    wiki.save_data(articles, "ml_articles.json")
    wiki.print_stats()

    # HuggingFace collection
    hf = HuggingFaceCollector()
    dataset = hf.download_dataset("squad", split="train[:1000]")

    print(f"\\nLoaded {len(dataset)} examples")


if __name__ == "__main__":
    example_collection()
\`\`\`

### Concurrent Collection

For faster collection with rate limiting:

\`\`\`python
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class ParallelCollector(DataCollector):
    """Collect data in parallel while respecting rate limits."""

    def __init__(self, output_dir: str, rate_limit: float = 1.0, max_workers: int = 5):
        super().__init__(output_dir, rate_limit)
        self.max_workers = max_workers

    def fetch_batch(self, urls: list) -> list:
        """Fetch multiple URLs in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.fetch, url): url
                for url in urls
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_url), total=len(urls)):
                url = future_to_url[future]
                try:
                    data = future.result()
                    results.append({'url': url, 'data': data})
                except Exception as e:
                    logger.error(f"Failed to fetch {url}: {e}")
                    results.append({'url': url, 'data': None, 'error': str(e)})

        return results
\`\`\`

### Best Practices

**Rate Limiting:**
- Respect robots.txt
- Add delays between requests
- Use exponential backoff on errors

**Caching:**
- Cache everything by default
- Use content-based hashing for cache keys
- Implement cache expiration if needed

**Error Handling:**
- Retry with exponential backoff
- Log all errors
- Continue on individual failures

**Storage:**
- Use structured formats (JSON, Parquet)
- Compress large files
- Organize by date/category

### Practice Exercises

- [ ] Build Wikipedia article collector
- [ ] Implement concurrent fetching
- [ ] Add progress bars
- [ ] Handle API pagination
- [ ] Implement cache expiration
- [ ] Add request throttling`
				}
			]
		},
		{
			name: 'Data Cleaning & Quality',
			description: 'Clean and validate data for machine learning',
			tasks: [
				{
					title: 'Implement comprehensive text cleaning pipeline',
					description: 'Build robust text cleaning with encoding fixes, HTML removal, and normalization',
					details: `## Text Cleaning Pipeline

### The Reality of Data Cleaning

**Time Breakdown:**
- Data Cleaning: 30% of time (most people think 5%)
- Data is always messy
- Clean data = better models

### Comprehensive Text Cleaner

\`\`\`python
import re
import unicodedata
from typing import List, Optional
import ftfy  # Fixes text encoding issues

class TextCleaner:
    """
    Comprehensive text cleaning pipeline.

    Handles:
    - Encoding issues (mojibake)
    - HTML tags
    - URLs and emails
    - Unicode normalization
    - Whitespace
    - Control characters
    """

    def __init__(self):
        # Compile regex patterns once for efficiency
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'[\\w\\.-]+@[\\w\\.-]+\\.\\w+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\\s+')
        self.repeated_chars_pattern = re.compile(r'(.)\\1{4,}')  # 5+ repeated chars

    def fix_encoding(self, text: str) -> str:
        """
        Fix encoding issues (mojibake, etc.).

        Examples:
        - "Ã©" → "é"
        - "â€™" → "'"
        """
        return ftfy.fix_text(text)

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters.

        NFKC: Compatibility decomposition + canonical composition
        - Converts different representations to standard form
        - "ﬁ" (ligature) → "fi"
        """
        return unicodedata.normalize('NFKC', text)

    def remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        return self.html_pattern.sub(' ', text)

    def remove_urls(self, text: str, replace_with: str = ' ') -> str:
        """Remove or replace URLs."""
        return self.url_pattern.sub(replace_with, text)

    def remove_emails(self, text: str) -> str:
        """Remove email addresses."""
        return self.email_pattern.sub(' ', text)

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace:
        - Multiple spaces → single space
        - Tabs → spaces
        - Strip leading/trailing
        """
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()

    def remove_repeated_chars(self, text: str, max_repeat: int = 2) -> str:
        """
        Remove excessive repeated characters.

        Example: "yesssssss" → "yess"
        """
        return self.repeated_chars_pattern.sub(r'\\1' * max_repeat, text)

    def remove_control_chars(self, text: str) -> str:
        """Remove control characters except newlines and tabs."""
        return ''.join(
            char for char in text
            if unicodedata.category(char) != 'Cc' or char in '\\n\\t'
        )

    def lowercase(self, text: str) -> str:
        """Convert to lowercase."""
        return text.lower()

    def remove_punctuation(self, text: str, keep: str = '') -> str:
        """
        Remove punctuation.

        Args:
            keep: string of punctuation to keep (e.g., ".,!?")
        """
        import string
        translator = str.maketrans('', '', ''.join(
            c for c in string.punctuation if c not in keep
        ))
        return text.translate(translator)

    def clean(self,
              text: str,
              fix_encoding: bool = True,
              normalize_unicode: bool = True,
              remove_html: bool = True,
              remove_urls: bool = True,
              remove_emails: bool = True,
              remove_control: bool = True,
              remove_repeated: bool = True,
              normalize_whitespace: bool = True,
              lowercase: bool = False,
              remove_punctuation: bool = False) -> str:
        """
        Apply full cleaning pipeline.

        Order matters - encoding fixes should come first.
        """
        if not text:
            return ""

        # Fix encoding first
        if fix_encoding:
            text = self.fix_encoding(text)

        # Unicode normalization
        if normalize_unicode:
            text = self.normalize_unicode(text)

        # Remove unwanted characters
        if remove_control:
            text = self.remove_control_chars(text)

        # Remove HTML
        if remove_html:
            text = self.remove_html(text)

        # Remove URLs and emails
        if remove_urls:
            text = self.remove_urls(text)

        if remove_emails:
            text = self.remove_emails(text)

        # Handle repeated characters
        if remove_repeated:
            text = self.remove_repeated_chars(text)

        # Case normalization
        if lowercase:
            text = self.lowercase(text)

        # Punctuation
        if remove_punctuation:
            text = self.remove_punctuation(text)

        # Whitespace normalization (do last)
        if normalize_whitespace:
            text = self.normalize_whitespace(text)

        return text


# Batch processing
class BatchTextCleaner:
    """Process large datasets efficiently."""

    def __init__(self, cleaner: TextCleaner = None):
        self.cleaner = cleaner or TextCleaner()

    def clean_batch(self, texts: List[str], **cleaning_options) -> List[str]:
        """Clean a batch of texts."""
        return [self.cleaner.clean(text, **cleaning_options) for text in texts]

    def clean_parallel(self, texts: List[str], n_jobs: int = 4, **cleaning_options) -> List[str]:
        """Clean texts in parallel using multiprocessing."""
        from multiprocessing import Pool
        from functools import partial

        clean_fn = partial(self.cleaner.clean, **cleaning_options)

        with Pool(n_jobs) as pool:
            cleaned = pool.map(clean_fn, texts)

        return cleaned

    def clean_dataset(self, dataset, text_column: str = 'text', **cleaning_options):
        """Clean a HuggingFace dataset."""
        def clean_example(example):
            example[text_column] = self.cleaner.clean(
                example[text_column],
                **cleaning_options
            )
            return example

        return dataset.map(clean_example, num_proc=4)


# Example usage
def example_cleaning():
    cleaner = TextCleaner()

    # Test cases
    test_texts = [
        # Encoding issue
        "Iâ€™m learning ML!",

        # HTML
        "<p>This is <strong>important</strong>!</p>",

        # URLs
        "Check out https://example.com for more info",

        # Repeated chars
        "Amazingggggg!!!!!!",

        # Mixed issues
        "  <div>Visit   http://test.com  </div>  ",
    ]

    print("=== Text Cleaning Examples ===\\n")

    for i, text in enumerate(test_texts, 1):
        cleaned = cleaner.clean(text)
        print(f"Example {i}:")
        print(f"  Before: {repr(text)}")
        print(f"  After:  {repr(cleaned)}")
        print()


# Statistics
def analyze_cleaning(texts: List[str], cleaner: TextCleaner):
    """Analyze cleaning impact."""
    cleaned = [cleaner.clean(text) for text in texts]

    original_length = sum(len(t) for t in texts)
    cleaned_length = sum(len(t) for t in cleaned)
    reduction = (1 - cleaned_length / original_length) * 100

    print(f"Original total length: {original_length:,}")
    print(f"Cleaned total length: {cleaned_length:,}")
    print(f"Reduction: {reduction:.1f}%")

    # Count empty after cleaning
    empty_after = sum(1 for t in cleaned if not t.strip())
    print(f"Empty after cleaning: {empty_after} ({empty_after/len(texts)*100:.1f}%)")


if __name__ == "__main__":
    example_cleaning()
\`\`\`

### Language-Specific Cleaning

**Code Cleaning:**
\`\`\`python
class CodeCleaner:
    """Clean code data specifically."""

    def remove_comments(self, code: str, language: str) -> str:
        """Remove comments from code."""
        patterns = {
            'python': r'#.*?$|\\'\\'\\'[\\s\\S]*?\\'\\'\\'\|\\"\\"\\"[\\s\\S]*?\\"\\"\\"',
            'javascript': r'//.*?$|/\\*[\\s\\S]*?\\*/',
            'java': r'//.*?$|/\\*[\\s\\S]*?\\*/',
        }

        pattern = re.compile(patterns.get(language, ''), re.MULTILINE)
        return pattern.sub('', code)

    def normalize_indentation(self, code: str, spaces: int = 4) -> str:
        """Convert tabs to spaces."""
        return code.replace('\\t', ' ' * spaces)
\`\`\`

### Practice Exercises

- [ ] Implement text cleaner from scratch
- [ ] Test on real messy data
- [ ] Add language detection
- [ ] Build cleaning report
- [ ] Compare before/after statistics
- [ ] Handle edge cases`
				}
			]
		}
	]
};

async function seed() {
	console.log('Seeding ML Pipeline path...');

	const pathResult = db.insert(schema.paths).values({
		name: mlPipelinePath.name,
		description: mlPipelinePath.description,
		color: mlPipelinePath.color,
		language: mlPipelinePath.language,
		skills: mlPipelinePath.skills,
		startHint: mlPipelinePath.startHint,
		difficulty: mlPipelinePath.difficulty,
		estimatedWeeks: mlPipelinePath.estimatedWeeks,
		schedule: mlPipelinePath.schedule
	}).returning().get();

	console.log(`Created path: ${mlPipelinePath.name}`);

	for (let i = 0; i < mlPipelinePath.modules.length; i++) {
		const mod = mlPipelinePath.modules[i];
		const moduleResult = db.insert(schema.modules).values({
			pathId: pathResult.id,
			name: mod.name,
			description: mod.description,
			orderIndex: i
		}).returning().get();

		console.log(`  Created module: ${mod.name}`);

		for (let j = 0; j < mod.tasks.length; j++) {
			const task = mod.tasks[j];
			db.insert(schema.tasks).values({
				moduleId: moduleResult.id,
				title: task.title,
				description: task.description,
				details: task.details,
				orderIndex: j,
				completed: false
			}).run();
		}
		console.log(`    Added ${mod.tasks.length} tasks`);
	}

	console.log('\nSeeding complete!');
}

seed().catch(console.error);
