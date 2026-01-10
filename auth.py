#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "huggingface-hub>=0.16.0",
# ]
# ///

"""
Authentication module for HuggingFace Hub
"""
import os
import sys
from typing import Optional

try:
    from huggingface_hub import HfApi, login, logout, whoami
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub not installed.")
    print("Run: pip install huggingface_hub")
    sys.exit(1)


class HFAuthManager:
    """Manages HuggingFace Hub authentication"""

    def __init__(self):
        self.api = HfApi()

    def is_authenticated(self) -> tuple[bool, Optional[str]]:
        """Check if user is authenticated
        
        Returns:
            (is_authenticated, username_or_error_message)
        """
        try:
            user_info = whoami()
            return True, user_info.get("name", "Unknown")
        except Exception as e:
            return False, str(e)

    def login_interactive(self, token: Optional[str] = None) -> bool:
        """Interactive login to HuggingFace Hub
        
        Args:
            token: Optional token string. If None, prompts user.
            
        Returns:
            True if login successful, False otherwise
        """
        print("\n" + "="*60)
        print("🔐 HuggingFace Hub Authentication")
        print("="*60)
        
        try:
            if token:
                login(token=token, write_permission=True)
                print("\n✅ Login successful using provided token")
            else:
                print("\nGet your access token at: https://huggingface.co/settings/tokens")
                print("Tip: Create a token with 'write' permission")
                login(write_permission=True)
                print("\n✅ Login successful")
            
            # Verify login
            user_info = whoami()
            print(f"   Logged in as: {user_info.get('name', 'Unknown')}")
            print("="*60 + "\n")
            return True
            
        except KeyboardInterrupt:
            print("\n\n❌ Login cancelled")
            return False
        except Exception as e:
            print(f"\n❌ Login failed: {e}")
            return False

    def logout(self) -> bool:
        """Logout from HuggingFace Hub
        
        Returns:
            True if logout successful, False otherwise
        """
        try:
            logout()
            print("\n✅ Logged out successfully")
            return True
        except Exception as e:
            print(f"\n❌ Logout failed: {e}")
            return False

    def show_status(self):
        """Show authentication status"""
        is_auth, result = self.is_authenticated()
        
        print("\n" + "="*60)
        print("📋 HuggingFace Authentication Status")
        print("="*60)
        
        if is_auth:
            print(f"✅ Status: Logged in")
            print(f"   User: {result}")
            print(f"   Token: ~/.cache/huggingface/token")
        else:
            print(f"❌ Status: Not logged in")
            print(f"   Error: {result}")
            print(f"\nLogin with: llmfit login")
        
        print("="*60 + "\n")

    def get_auth_error_message(self, model_name: Optional[str] = None) -> str:
        """Get user-friendly error message for authentication failures
        
        Args:
            model_name: Name of the model that required authentication
            
        Returns:
            Formatted error message
        """
        msg = "\n" + "="*60
        msg += "\n❌ Authentication Required\n"
        msg += "="*60
        
        if model_name:
            msg += f"\nAccess to '{model_name}' requires a HuggingFace account.\n"
        else:
            msg += "\nThis operation requires HuggingFace authentication.\n"
        
        msg += "\nSign in options:\n"
        msg += "  1. Interactive login: llmfit login\n"
        msg += "  2. CLI command:      huggingface-cli login\n"
        msg += "  3. Set environment:  export HF_TOKEN=your_token\n"
        
        msg += "\nGet your token at: https://huggingface.co/settings/tokens"
        msg += "\nCreate a token with 'write' permission for full access"
        msg += "\n\nAfter signing in, retry your command."
        msg += "\n" + "="*60 + "\n"
        
        return msg


def check_and_handle_auth_error(func):
    """Decorator to check for HuggingFace authentication errors
    
    Catches HfHubHTTPError with 401/403 status and displays helpful message
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HfHubHTTPError as e:
            if e.response.status_code in (401, 403):
                auth_manager = HFAuthManager()
                # Try to extract model name from context if available
                model_name = kwargs.get('model_name') or getattr(args[0], '__name__', 'this resource')
                print(auth_manager.get_auth_error_message(model_name))
                sys.exit(1)
            raise
    return wrapper
