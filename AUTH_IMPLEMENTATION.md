# HuggingFace Authentication Implementation

## Summary
Added HuggingFace Hub authentication support to `llmfit` with a modular `auth.py` module.

## New Features

### CLI Commands

1. **`llmfit login`** - Interactive sign-in to HuggingFace Hub
   - Prompts for HuggingFace token
   - Requires 'write' permission token
   - Stores token at `~/.cache/huggingface/token`

2. **`llmfit logout`** - Sign out from HuggingFace Hub
   - Removes stored token

3. **`llmfit status`** - Check authentication status
   - Shows if logged in and username

## Error Handling

When authentication is required (401/403 errors):

```
============================================================
❌ Authentication Required
============================================================

Access to 'meta-llama/Llama-3.1-70B-Instruct' requires a HuggingFace account.

Sign in options:
  1. Interactive login: llmfit login
  2. CLI command:      huggingface-cli login
  3. Set environment:  export HF_TOKEN=your_token

Get your token at: https://huggingface.co/settings/tokens
Create a token with 'write' permission for full access

After signing in, retry your command.
============================================================
```

## Architecture

### Files Created
- **`auth.py`** - HFAuthManager class handling:
  - Authentication status checking (`is_authenticated()`)
  - Interactive login (`login_interactive()`)
  - Logout (`logout()`)
  - Status display (`show_status()`)
  - User-friendly error messages (`get_auth_error_message()`)

### Files Modified
- **`llmfit.py`** - Integrated auth module:
  - Added import for HFAuthManager
  - Added login/logout/status subcommands
  - Wrapped `AutoConfig.from_pretrained()` with auth error detection
  - Updated error handling in `list_providers()` and `list_models()`
  - Updated help text with new commands

## Usage Examples

```bash
# Check authentication status
llmfit status

# Login to HuggingFace (follow prompts)
llmfit login

# After login, access gated models
llmfit estimate meta-llama/Llama-3.1-70B-Instruct

# Logout when done
llmfit logout
```

## Token Storage
- Location: `~/.cache/huggingface/token`
- Used by all HuggingFace Hub operations
- Persistent across sessions
- Compatible with `huggingface-cli login`

## Compatibility
- Works with existing symlink workflow (`~/.local/bin/llmfit`)
- Token storage is compatible with official HuggingFace CLI tools
- No breaking changes to existing functionality

## Future Enhancements
- Could add `llmfit login --token <token>` for non-interactive login
- Could extract `LLMMemoryEstimator` to separate `estimator.py` module
- Could extract browsing logic to separate `browser.py` module
