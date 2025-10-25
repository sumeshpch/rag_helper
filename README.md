# RAG Helper - Retrieval-Augmented Generation

A simple yet powerful Retrieval-Augmented Generation (RAG) implementation that combines semantic search with AI language models to answer questions using your own knowledge base.

## Features

- 🔍 **Semantic Retrieval**: Finds relevant context using embeddings
- 🤖 **Dual AI Options**: Use local HuggingFace models (free) OR OpenAI GPT (paid, higher quality)
- ⚡ **Fast & Efficient**: Lightweight embedding model for quick searches
- 🆓 **Free Option**: Run completely offline with local models (no API costs!)
- 📚 **Custom Knowledge Base**: Easy to add your own documents
- 💡 **Interactive**: Command-line interface for asking questions
- 🔒 **Privacy**: Local mode keeps all data on your machine

## How It Works

RAG (Retrieval-Augmented Generation) combines two powerful techniques:

1. **Retrieval**: 
   - Embeds your documents and the query into vector space
   - Finds the most relevant documents using cosine similarity
   - Retrieves top-K most similar documents as context

2. **Generation**:
   - Sends the query + retrieved context to an AI model (local or cloud)
   - Generates an accurate, context-aware answer
   - Only uses information from your knowledge base

This approach ensures the AI provides accurate, domain-specific answers grounded in your data.

## Two Modes of Operation

### 🆓 Local Mode (Free)
- Uses HuggingFace models (GPT-2, DistilGPT2, etc.)
- Runs completely offline
- No API costs
- Privacy-friendly (all data stays local)
- Lower quality answers compared to GPT-4

### ☁️ Cloud Mode (Paid)
- Uses OpenAI GPT models (GPT-4o-mini, GPT-4, etc.)
- Requires API key and internet
- Small cost per query (~$0.00015 for GPT-4o-mini)
- Significantly better answer quality
- Requires OpenAI account

### Quick Comparison

| Feature | Local Mode | Cloud Mode |
|---------|-----------|------------|
| **Cost** | Free | ~$0.00015/query |
| **Quality** | Good | Excellent |
| **Privacy** | 100% Private | Sent to OpenAI |
| **Speed** | Fast | Very Fast |
| **Setup** | No API key | Needs API key |
| **Internet** | Not required | Required |
| **Best For** | Testing, privacy, cost | Production, quality |

## Requirements

- Python 3.11 or 3.12
- OpenAI API key (only for Cloud Mode)
- Dependencies listed in `requirements.txt`

## Installation

### 1. Navigate to Project Directory

```bash
cd /Users/sumesh/python/rag_helper
```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Choose Your Mode

The script runs in **Local Mode** by default (free, no API key needed).

#### Option A: Use Local Mode (Default - Free)
No setup needed! Just run the script. It will use GPT-2 locally.

#### Option B: Use Cloud Mode (Better Quality)
Set your OpenAI API key:

```bash
# Set environment variable
export OPENAI_API_KEY="your_openai_api_key_here"

# Or add to your shell profile (~/.zshrc or ~/.bashrc)
echo 'export OPENAI_API_KEY="your_openai_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

Then edit `main.py` and change:
```python
USE_HUGGINGFACE = False  # Switch to OpenAI
```

**Get your API key**: [OpenAI Platform](https://platform.openai.com/api-keys)

## Usage

### Quick Start (Local Mode - Free)

Just run the script:

```bash
python main.py
```

You'll be prompted:
```
Enter your question: 
```

Type your question and press Enter. The system will:
1. Find relevant documents from the knowledge base
2. Generate an answer using the local GPT-2 model
3. Display the results

**First run**: Downloads GPT-2 model (~548MB), subsequent runs are instant.

### Example Session (Local Mode)

```
🔹 Loading embedding model...
Enter your question: what are type resolvers in magento graphql

🔹 Retrieved Contexts:
 - A type resolver in GraphQL maps a schema type to its backend data source.
 - GraphQL in Magento allows efficient data fetching for headless frontends.
 - Vector databases store and retrieve high-dimensional embeddings for similarity search.

🔹 Generating answer...
🔹 Loading HuggingFace model: gpt2

💬 Answer:
Type resolvers in Magento GraphQL map schema types to their backend data sources, 
enabling efficient data fetching for headless frontends.
```

### Example Session (Cloud Mode with OpenAI)

```
🔹 Loading embedding model...
Enter your question: What is RAG and how is it used?

🔹 Retrieved Contexts:
 - RAG combines data retrieval and generative AI to answer domain-specific questions.
 - Vector databases store and retrieve high-dimensional embeddings for similarity search.
 - OpenAI models like GPT-4 can generate human-like text when given contextual information.

🔹 Generating answer...

💬 Answer:
RAG (Retrieval-Augmented Generation) is a technique that combines data retrieval 
with generative AI to answer domain-specific questions. It works by first retrieving 
relevant information from a knowledge base using vector similarity search, then using 
that context to generate accurate, grounded responses with AI models like GPT-4.
```

## Customizing Your Knowledge Base

### Add Your Own Documents

Edit the `DOCUMENTS` list in `main.py`:

```python
DOCUMENTS = [
    "Your first piece of knowledge here",
    "Another important fact or document",
    "Add as many documents as you need",
    # ...
]
```

### Configuration Options

Modify the configuration section in `main.py`:

```python
MODEL_EMB = "all-MiniLM-L6-v2"  # Embedding model
GPT_MODEL = "gpt-4o-mini"       # OpenAI GPT model
HUGGINGFACE_MODEL = "gpt2"      # Local HuggingFace model
TOP_K = 3                       # Number of documents to retrieve
USE_HUGGINGFACE = True          # True = Local, False = OpenAI
```

### Available Local Models (HuggingFace)

Free, no authentication required:
- `gpt2` - Medium quality, 548MB (recommended for local)
- `distilgpt2` - Fast but basic quality, 82MB
- `gpt2-medium` - Better quality, 1.5GB
- `EleutherAI/gpt-neo-125M` - Decent quality, 125M params

**Note**: First run downloads the model, subsequent runs are instant.

### Available Cloud Models (OpenAI)

Requires API key and costs per query:
- `gpt-4o-mini` - Fast and cost-effective (~$0.00015/query, recommended)
- `gpt-4o` - Most capable (~$0.003/query)
- `gpt-4-turbo` - Good balance of speed and capability
- `gpt-3.5-turbo` - Fastest, lowest cost

### Alternative Embedding Models

- `all-MiniLM-L6-v2` - Fast, lightweight (default, recommended)
- `all-mpnet-base-v2` - Higher quality, slower
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

## Project Structure

```
rag_helper/
├── main.py              # Main RAG implementation
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── venv/               # Virtual environment (not tracked)
```

## API Reference

### Key Functions

#### `retrieve_context(query, top_k=TOP_K)`
Finds the most relevant documents for a given query.

**Parameters:**
- `query` (str): The question or search query
- `top_k` (int): Number of documents to retrieve (default: 3)

**Returns:** List of relevant document strings

**Example:**
```python
context = retrieve_context("What is GraphQL?", top_k=3)
```

#### `generate_answer(query, context_docs)`
Generates an answer using GPT with the provided context.

**Parameters:**
- `query` (str): The user's question
- `context_docs` (list): List of relevant document strings

**Returns:** Generated answer as a string

**Example:**
```python
answer = generate_answer("What is GraphQL?", context_docs)
print(answer)
```

## Cost Comparison

### Local Mode (HuggingFace)
- **Cost**: $0 (completely free!)
- **Quality**: Moderate (decent for simple Q&A)
- **Speed**: Fast after initial model download
- **Privacy**: 100% private, runs offline

### Cloud Mode (OpenAI)

Using `gpt-4o-mini`:
- **Cost**: ~$0.00015 per question (varies with context length)
- **Quality**: Excellent, human-like responses
- **Speed**: Very fast (depends on internet)
- Very affordable for personal projects (~6,600 questions per $1)

Using `gpt-4o`:
- **Cost**: ~$0.002-0.005 per question
- **Quality**: Best available
- Higher quality, more nuanced responses

**Recommendation**: Start with local mode for testing, switch to OpenAI for production or when you need high-quality answers.

## Troubleshooting

### "Repetitive or Low-Quality Answers" (Local Mode)

If local models generate repetitive text:
- Switch to `gpt2` instead of `distilgpt2` (better quality)
- Try `gpt2-medium` for even better results (1.5GB)
- Consider switching to OpenAI for production use

### "OpenAI API Key Not Found" (Cloud Mode)

```bash
# Make sure the key is exported
echo $OPENAI_API_KEY

# If empty, set it:
export OPENAI_API_KEY="your_key_here"

# Or switch to local mode in main.py:
USE_HUGGINGFACE = True
```

### "Model Download Taking Too Long" (Local Mode)

First run downloads the model:
- `distilgpt2`: ~82MB (< 1 minute)
- `gpt2`: ~548MB (2-5 minutes)
- `gpt2-medium`: ~1.5GB (5-10 minutes)

Models are cached and subsequent runs are instant.

### Rate Limit Errors (Cloud Mode)

If you hit OpenAI rate limits:
- Switch to local mode (no rate limits!)
- Add a delay between requests
- Upgrade your OpenAI plan
- Use `gpt-3.5-turbo` for higher rate limits

### Python Version Issues

Ensure you're using Python 3.11 or 3.12:

```bash
python3 --version
```

### NumPy Version Conflicts

If you encounter NumPy errors:

```bash
pip install "numpy<2"
```

## Advanced Usage

### Save Embeddings for Faster Startup

```python
import pickle
import numpy as np

# Save embeddings
np.save("doc_embeddings.npy", doc_embeddings)
with open("documents.pkl", "wb") as f:
    pickle.dump(DOCUMENTS, f)

# Load embeddings
doc_embeddings = np.load("doc_embeddings.npy")
with open("documents.pkl", "rb") as f:
    DOCUMENTS = pickle.load(f)
```

### Batch Processing Multiple Questions

```python
questions = [
    "What is Adobe Commerce?",
    "How does GraphQL work?",
    "What is RAG?"
]

for q in questions:
    context = retrieve_context(q)
    answer = generate_answer(q, context)
    print(f"\nQ: {q}\nA: {answer}\n")
```

### Custom Prompts

Modify the prompt in `generate_answer()` for different behaviors:

```python
prompt = f"""
You are a technical expert specializing in eCommerce and web development.

Context:
{context}

Question: {query}

Provide a detailed technical answer with examples.
"""
```

## Use Cases

- 📚 **Documentation Q&A**: Answer questions about your product docs
- 💼 **Internal Knowledge Base**: Help employees find company information
- 🎓 **Educational Assistant**: Create learning tools with custom content
- 🔧 **Technical Support**: Automated support using your help articles
- 📊 **Research Assistant**: Query research papers and reports
- 🏢 **Customer Support**: Answer customer questions using your FAQ

## Scaling Up

For larger knowledge bases, consider:

1. **Vector Databases**: Use Pinecone, Weaviate, or ChromaDB
2. **Better Embeddings**: Try OpenAI's `text-embedding-3-small` or `text-embedding-3-large`
3. **Document Chunking**: Split large documents into smaller chunks
4. **Hybrid Search**: Combine semantic search with keyword matching
5. **Caching**: Cache frequent queries to reduce API costs

## Security Best Practices

- ✅ Never commit API keys to git
- ✅ Use environment variables for sensitive data
- ✅ Rotate API keys regularly
- ✅ Monitor API usage and costs
- ✅ Implement rate limiting in production

## Performance Tips

### For Local Mode (HuggingFace)
- Use `gpt2` for best balance of speed and quality
- Use `distilgpt2` if you need maximum speed
- Reduce `max_new_tokens` to speed up generation
- Models are cached after first download

### For Cloud Mode (OpenAI)
- Use `gpt-4o-mini` for faster, cheaper responses
- Use `gpt-4o` when you need highest quality
- Reduce `TOP_K` if context is too long
- Cache embeddings to avoid recomputation
- Use batch processing for multiple queries

### General
- Cache document embeddings (see Advanced Usage)
- Use smaller embedding models if memory is limited
- Process queries in batches when possible

## Related Projects

- **semantic_search**: Local-only semantic search (no API required)
- [LangChain](https://github.com/langchain-ai/langchain): Full-featured RAG framework
- [LlamaIndex](https://github.com/run-llama/llama_index): Data framework for LLM apps

## License

MIT License - Feel free to use this code for your projects!

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Acknowledgments

- [OpenAI](https://openai.com/) - GPT models and APIs
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [HuggingFace](https://huggingface.co/) - Model hosting
- [scikit-learn](https://scikit-learn.org/) - Similarity calculations

---

**Happy RAG Building! 🚀**

