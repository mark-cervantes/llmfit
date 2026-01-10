#!/home/cmark/.local/bin/llmfit-dir/.venv/bin/python3
"""
llmfit - Estimate LLM memory requirements and browse models
"""
import argparse
import re
import sys
from typing import Optional, List, Dict
from dataclasses import dataclass

try:
    from huggingface_hub import HfApi, list_models
    from transformers import AutoConfig
    import torch
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
    from prompt_toolkit.shortcuts import radiolist_dialog, button_dialog
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install torch transformers huggingface_hub prompt-toolkit")
    sys.exit(1)

from auth import HFAuthManager


@dataclass
class ModelInfo:
    name: str
    params_b: float
    provider: str
    tags: List[str]


class LLMMemoryEstimator:
    PRECISION_MAP = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "4bit": 0.5,
        "2bit": 0.25,
    }

    def __init__(self):
        self.api = HfApi()

    def list_providers(self, task: str = "text-generation", limit: int = 50, interactive: bool = False):
        """List ALL model providers/organizations from HuggingFace"""
        print(f"\n🔍 Fetching providers (task={task})...")
        
        try:
            # Fetch models and extract unique providers
            model_iterator = list_models(
                filter=task,
                sort="downloads",
                direction=-1,
                limit=500,  # Fetch more to get diverse providers
            )
            
            # Count models per provider
            provider_counts = {}
            for model in model_iterator:
                provider = model.id.split("/")[0] if "/" in model.id else model.id
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            
            # Sort by model count (popularity)
            sorted_providers = sorted(
                provider_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
            
            providers = [p[0] for p in sorted_providers]
            
            if interactive:
                # Interactive selection
                return self._interactive_provider_select(sorted_providers, task)
            else:
                # Just list them
                print(f"\nAll LLM Providers (task={task}, top {len(sorted_providers)}):")
                print(f"{'Provider':<30} {'Models':>10}")
                print("─" * 42)
                
                for provider, count in sorted_providers:
                    print(f"{provider:<30} {count:>10}")
                
                print(f"\n✅ Found {len(providers)} providers")
                print(f"Use: llmfit list <provider> to see models")
                print(f"Or: llmfit browse for interactive mode")
                return providers
            
        except Exception as e:
            error_str = str(e)
            # Check for authentication errors
            if any(code in error_str for code in ['401', '403', 'Unauthorized', 'Forbidden']):
                auth_manager = HFAuthManager()
                print(auth_manager.get_auth_error_message())
                return []
            print(f"❌ Error fetching providers: {e}")
            # Fallback to hardcoded list
            print("\nFalling back to known providers:")
            popular = [
                "meta-llama", "Qwen", "mistralai", "google", "microsoft",
                "EleutherAI", "stabilityai", "tiiuae", "01-ai", "bigscience",
                "bigcode", "Salesforce", "databricks", "upstage", "NousResearch",
                "teknium", "Open-Orca", "TheBloke", "bartowski", "lmstudio-ai",
            ]
            for i, org in enumerate(popular, 1):
                print(f"  {i:2}. {org}")
            return popular

    def _interactive_provider_select(self, providers_with_counts, task):
        """Interactive provider selection with fuzzy search"""
        try:
            # Create fuzzy searchable list
            provider_names = [p[0] for p in providers_with_counts]
            
            print(f"\n{'='*60}")
            print(f"🔍 Interactive Provider Browser")
            print(f"{'='*60}")
            print("Type to search, press Enter to select")
            print("─" * 60)
            
            completer = FuzzyCompleter(WordCompleter(provider_names, ignore_case=True))
            selected = prompt(
                "Select provider (fuzzy search): ",
                completer=completer,
                complete_while_typing=True,
            )
            
            if selected and selected in provider_names:
                print(f"\n✅ Selected: {selected}")
                return selected
            else:
                print(f"\n❌ Invalid selection")
                return None
                
        except (KeyboardInterrupt, EOFError):
            print("\n\n❌ Cancelled")
            return None

    def browse_interactive(self, task: str = "text-generation"):
        """Interactive browsing experience"""
        print(f"\n{'='*60}")
        print(f"🚀 LLMFit Interactive Browser")
        print(f"{'='*60}\n")
        
        # Step 1: Select provider
        provider = self.list_providers(task=task, interactive=True)
        if not provider:
            return
        
        # Step 2: Get models from provider
        print(f"\n🔍 Fetching models from {provider}...")
        models = self.list_models(provider=provider, limit=50, task=task)
        
        if not models:
            print(f"❌ No models found for {provider}")
            return
        
        # Step 3: Select model interactively
        selected_model = self._interactive_model_select(models)
        if not selected_model:
            return
        
        # Step 4: Configure estimation
        precision = self._interactive_precision_select()
        if not precision:
            return
        
        # Step 5: Run estimation
        print(f"\n🎯 Estimating memory for {selected_model.name}...")
        self.estimate_memory(
            model_name=selected_model.name,
            precision=precision,
            context_length=4096,
            batch_size=1,
        )

    def _interactive_model_select(self, models: List[ModelInfo]):
        """Interactive model selection with fuzzy search"""
        try:
            model_names = [m.name for m in models]
            model_dict = {m.name: m for m in models}
            
            print(f"\n{'='*60}")
            print(f"🔍 Model Browser ({len(models)} models)")
            print(f"{'='*60}")
            print("Type to search, press Enter to select")
            print("─" * 60)
            
            completer = FuzzyCompleter(WordCompleter(model_names, ignore_case=True))
            selected = prompt(
                "Select model (fuzzy search): ",
                completer=completer,
                complete_while_typing=True,
            )
            
            if selected and selected in model_dict:
                model = model_dict[selected]
                print(f"\n✅ Selected: {model.name} ({model.params_b:.1f}B)")
                return model
            else:
                print(f"\n❌ Invalid selection")
                return None
                
        except (KeyboardInterrupt, EOFError):
            print("\n\n❌ Cancelled")
            return None

    def _interactive_precision_select(self):
        """Interactive precision selection"""
        try:
            precisions = list(self.PRECISION_MAP.keys())
            
            print(f"\n{'='*60}")
            print(f"⚙️  Select Quantization Precision")
            print(f"{'='*60}")
            print("Type to search, press Enter to select")
            print("─" * 60)
            
            completer = FuzzyCompleter(WordCompleter(precisions, ignore_case=True))
            selected = prompt(
                "Select precision (default: 4bit): ",
                completer=completer,
                complete_while_typing=True,
                default="4bit",
            )
            
            if selected and selected in precisions:
                print(f"\n✅ Selected: {selected}")
                return selected
            else:
                print(f"\n❌ Invalid selection, using 4bit")
                return "4bit"
                
        except (KeyboardInterrupt, EOFError):
            print("\n\n❌ Cancelled, using 4bit")
            return "4bit"

    def list_models(
        self,
        provider: Optional[str] = None,
        max_params: Optional[float] = None,
        min_params: Optional[float] = None,
        limit: int = 20,
        task: str = "text-generation",
    ) -> List[ModelInfo]:
        """List models from provider with optional size filtering"""
        filters = {"task": task}
        if provider:
            filters["author"] = provider

        print(f"\n🔍 Searching models (provider={provider or 'all'}, task={task})...")
        
        models = []
        try:
            model_iterator = list_models(
                filter=filters["task"],
                author=filters.get("author"),
                sort="downloads",
                direction=-1,
                limit=limit * 3,  # Fetch extra for filtering
            )
            
            for model in model_iterator:
                # Extract param count from model name/card
                params_b = self._extract_params(model.id, model.tags)
                if params_b is None:
                    continue
                
                # Apply size filters
                if max_params and params_b > max_params:
                    continue
                if min_params and params_b < min_params:
                    continue
                
                models.append(ModelInfo(
                    name=model.id,
                    params_b=params_b,
                    provider=model.id.split("/")[0],
                    tags=model.tags or [],
                ))
                
                if len(models) >= limit:
                    break
        
        except Exception as e:
            error_str = str(e)
            # Check for authentication errors
            if any(code in error_str for code in ['401', '403', 'Unauthorized', 'Forbidden']):
                auth_manager = HFAuthManager()
                print(auth_manager.get_auth_error_message())
                return []
            print(f"❌ Error fetching models: {e}")
            return []

        return models

    def _extract_params(self, model_id: str, tags: List[str]) -> Optional[float]:
        """Extract parameter count from model name or tags"""
        # Common patterns: "7B", "7b", "7B-", "7.2B", "70b-instruct"
        patterns = [
            r'[-_](\d+\.?\d*)[Bb](?:[-_]|$)',  # dash/underscore separated
            r'[-_](\d+\.?\d*)b(?:[-_]|$)',     # lowercase b
            r'^(\d+\.?\d*)[Bb][-_]',            # starts with number
            r'(\d+\.?\d*)[Bb]$',                # ends with number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_id)
            if match:
                return float(match.group(1))
        
        # Check tags (e.g., "7b", "70b")
        if tags:
            for tag in tags:
                match = re.match(r'^(\d+\.?\d*)[Bb]$', tag)
                if match:
                    return float(match.group(1))
        
        return None

    def estimate_memory(
        self,
        model_name: str,
        precision: str = "4bit",
        context_length: int = 4096,
        batch_size: int = 1,
    ) -> Dict:
        """Estimate memory requirements for a model"""
        if precision not in self.PRECISION_MAP:
            raise ValueError(f"Precision must be one of: {list(self.PRECISION_MAP.keys())}")
        
        bytes_per_param = self.PRECISION_MAP[precision]
        
        print(f"\nLoading config for: {model_name}")
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            error_str = str(e)
            # Check for authentication errors (401/403)
            if any(code in error_str for code in ['401', '403', 'Unauthorized', 'Forbidden']):
                auth_manager = HFAuthManager()
                print(auth_manager.get_auth_error_message(model_name))
                return {}
            print(f"❌ Failed to load config: {e}")
            return {}
        
        # Get param count from config or estimate
        num_params = self._estimate_params_from_config(config)
        weights_gb = (num_params * bytes_per_param) / 1e9
        
        # KV cache estimation (decoder-only models)
        kv_gb = 0
        if hasattr(config, 'num_hidden_layers') and hasattr(config, 'hidden_size'):
            n_layers = config.num_hidden_layers
            hidden_size = config.hidden_size
            
            # For multi-query/grouped-query attention
            if hasattr(config, 'num_key_value_heads'):
                kv_heads = config.num_key_value_heads
            elif hasattr(config, 'num_attention_heads'):
                kv_heads = config.num_attention_heads
            else:
                kv_heads = hidden_size // 64
            
            head_dim = hidden_size // (config.num_attention_heads if hasattr(config, 'num_attention_heads') else kv_heads)
            kv_bytes = n_layers * kv_heads * head_dim * context_length * 2 * bytes_per_param * batch_size
            kv_gb = kv_bytes / 1e9
        
        # Framework overhead (activations, gradients buffer, etc.)
        overhead_gb = max(1.5, weights_gb * 0.15)  # Min 1.5GB for PyTorch
        
        total_gb = weights_gb + kv_gb + overhead_gb
        
        result = {
            "model": model_name,
            "params_b": num_params / 1e9,
            "precision": precision,
            "context_length": context_length,
            "weights_gb": weights_gb,
            "kv_cache_gb": kv_gb,
            "overhead_gb": overhead_gb,
            "total_gb": total_gb,
            "fits_16gb": total_gb < 14,
            "fits_32gb": total_gb < 28,
            "fits_64gb": total_gb < 58,
        }
        
        self._print_estimate(result)
        return result

    def _estimate_params_from_config(self, config) -> int:
        """Estimate parameter count from model config"""
        # Check if directly available
        if hasattr(config, 'num_parameters'):
            return config.num_parameters
        
        # Estimate for transformer models
        if not all(hasattr(config, attr) for attr in ['num_hidden_layers', 'hidden_size']):
            raise ValueError("Cannot estimate params: missing config attributes")
        
        n_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        vocab_size = getattr(config, 'vocab_size', 32000)
        intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
        
        # Rough formula for decoder-only transformers (Llama-style)
        # Embedding: vocab_size * hidden_size
        # Per layer: attention (4 * hidden_size^2) + FFN (3 * hidden_size * intermediate_size)
        # Output head: vocab_size * hidden_size (often tied to embedding)
        
        embed_params = vocab_size * hidden_size
        attn_params_per_layer = 4 * hidden_size * hidden_size
        ffn_params_per_layer = 3 * hidden_size * intermediate_size
        layer_params = (attn_params_per_layer + ffn_params_per_layer) * n_layers
        
        total_params = embed_params + layer_params + vocab_size * hidden_size
        return int(total_params)

    def _print_estimate(self, result: Dict):
        """Pretty print estimation results"""
        print(f"\n{'='*60}")
        print(f"Model: {result['model']}")
        print(f"{'='*60}")
        print(f"Parameters:       {result['params_b']:.2f}B")
        print(f"Precision:        {result['precision']}")
        print(f"Context Length:   {result['context_length']:,} tokens")
        print(f"\n{'Memory Breakdown:':─<60}")
        print(f"  Weights:        {result['weights_gb']:>6.2f} GB")
        print(f"  KV Cache:       {result['kv_cache_gb']:>6.2f} GB")
        print(f"  Overhead:       {result['overhead_gb']:>6.2f} GB")
        print(f"  {'─'*56}")
        print(f"  TOTAL:          {result['total_gb']:>6.2f} GB")
        print(f"\n{'Compatibility:':─<60}")
        print(f"  16GB RAM:  {'✅ YES' if result['fits_16gb'] else '❌ NO (quantize further)'}")
        print(f"  32GB RAM:  {'✅ YES' if result['fits_32gb'] else '❌ NO'}")
        print(f"  64GB RAM:  {'✅ YES' if result['fits_64gb'] else '❌ NO'}")
        print(f"{'='*60}\n")


def parse_size_filter(size_str: str) -> tuple[Optional[float], Optional[float]]:
    """Parse size filters like '<7', '>=13', '7-13', '7b'"""
    size_str = size_str.lower().replace('b', '').strip()
    
    # Range: 7-13
    if '-' in size_str and not size_str.startswith('-'):
        parts = size_str.split('-')
        return float(parts[0]), float(parts[1])
    
    # Comparison operators
    if size_str.startswith('>='):
        return float(size_str[2:]), None
    elif size_str.startswith('<='):
        return None, float(size_str[2:])
    elif size_str.startswith('>'):
        return float(size_str[1:]) + 0.01, None  # Exclusive
    elif size_str.startswith('<'):
        return None, float(size_str[1:]) - 0.01  # Exclusive
    elif size_str.startswith('='):
        val = float(size_str[1:])
        return val - 0.5, val + 0.5  # Tolerance
    
    # Exact value
    try:
        val = float(size_str)
        return val - 0.5, val + 0.5  # Tolerance
    except ValueError:
        raise ValueError(f"Invalid size filter: {size_str}. Use: <7, >=13, 7-13, or 7")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate LLM memory requirements and browse models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive browser (like fzf)
  llmfit browse
  
  # Login to HuggingFace (required for gated models)
  llmfit login
  
  # Check authentication status
  llmfit status
  
  # List popular providers
  llmfit providers
  
  # List models from Qwen under 7B
  llmfit list Qwen --size "<7"
  
  # List all models between 7-13B
  llmfit list --size "7-13" --limit 30
  
  # Estimate memory for a specific model
  llmfit estimate Qwen/Qwen2.5-Coder-7B-Instruct --precision 4bit --context 8192
  
  # Quick estimate with defaults (4bit, 4k context)
  llmfit estimate meta-llama/Llama-3.2-3B-Instruct
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Browse command (interactive mode)
    browse_parser = subparsers.add_parser('browse', help='Interactive browser (like fzf)')
    browse_parser.add_argument('--task', default='text-generation', help='Model task (default: text-generation)')
    
    # Providers command
    subparsers.add_parser('providers', help='List popular model providers')
    
    # Login command
    subparsers.add_parser('login', help='Sign in to HuggingFace Hub')
    
    # Logout command
    subparsers.add_parser('logout', help='Sign out from HuggingFace Hub')
    
    # Status command
    subparsers.add_parser('status', help='Check authentication status')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List models from provider')
    list_parser.add_argument('provider', nargs='?', help='Provider name (e.g., Qwen, meta-llama)')
    list_parser.add_argument('--size', help='Size filter: <7, >=13, 7-13, =7')
    list_parser.add_argument('--limit', type=int, default=20, help='Max results (default: 20)')
    list_parser.add_argument('--task', default='text-generation', help='Model task (default: text-generation)')
    
    # Estimate command
    est_parser = subparsers.add_parser('estimate', help='Estimate memory for a model')
    est_parser.add_argument('model', help='Model name (e.g., Qwen/Qwen2.5-Coder-7B)')
    est_parser.add_argument('--precision', default='4bit', 
                           choices=['fp32', 'fp16', 'bf16', 'int8', '4bit', '2bit'],
                           help='Quantization precision (default: 4bit)')
    est_parser.add_argument('--context', type=int, default=4096, help='Context length (default: 4096)')
    est_parser.add_argument('--batch', type=int, default=1, help='Batch size (default: 1)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    estimator = LLMMemoryEstimator()
    
    if args.command == 'browse':
        estimator.browse_interactive(task=args.task)
    
    elif args.command == 'providers':
        estimator.list_providers()
    
    elif args.command == 'login':
        auth_manager = HFAuthManager()
        success = auth_manager.login_interactive()
        sys.exit(0 if success else 1)
    
    elif args.command == 'logout':
        auth_manager = HFAuthManager()
        success = auth_manager.logout()
        sys.exit(0 if success else 1)
    
    elif args.command == 'status':
        auth_manager = HFAuthManager()
        auth_manager.show_status()
    
    elif args.command == 'list':
        min_params, max_params = None, None
        if args.size:
            try:
                min_params, max_params = parse_size_filter(args.size)
            except ValueError as e:
                print(f"❌ {e}")
                sys.exit(1)
        
        models = estimator.list_models(
            provider=args.provider,
            max_params=max_params,
            min_params=min_params,
            limit=args.limit,
            task=args.task,
        )
        
        if not models:
            print("❌ No models found matching criteria")
            sys.exit(1)
        
        print(f"\n{'Model':<50} {'Size':>8} {'Provider':<20}")
        print("─" * 80)
        for m in models:
            print(f"{m.name:<50} {m.params_b:>6.1f}B {m.provider:<20}")
        
        print(f"\n✅ Found {len(models)} models")
        print("Use: llmfit estimate <model_name> to check memory")
    
    elif args.command == 'estimate':
        estimator.estimate_memory(
            model_name=args.model,
            precision=args.precision,
            context_length=args.context,
            batch_size=args.batch,
        )


if __name__ == "__main__":
    main()
