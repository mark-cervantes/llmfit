# llmfit

Estimate LLM memory requirements and browse models from HuggingFace.

## Setup

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -e .

# Verify installation
llmfit --help
```

## Usage

### Interactive Browser
```bash
llmfit browse
```

### Authentication
```bash
llmfit login      # Sign in to HuggingFace
llmfit status     # Check auth status
llmfit logout     # Sign out
```

### List Models
```bash
llmfit providers                # List popular providers
llmfit list Qwen               # List models from provider
llmfit list --size "<7"        # Filter by size
```

### Estimate Memory
```bash
llmfit estimate Qwen/Qwen2.5-Coder-7B-Instruct
llmfit estimate meta-llama/Llama-3.2-3B-Instruct --precision 4bit --context 8192
```
