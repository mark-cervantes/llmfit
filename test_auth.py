#!/usr/bin/env python3

"""
Test script for HuggingFace authentication module
"""
import sys
sys.path.insert(0, '/home/cmark/.local/bin/llmfit-dir')

from auth import HFAuthManager

print("Testing HFAuthManager...\n")

# Test 1: Authentication status
print("Test 1: Checking authentication status")
auth_manager = HFAuthManager()
is_auth, result = auth_manager.is_authenticated()
print(f"  Authenticated: {is_auth}")
print(f"  Result: {result}")
print()

# Test 2: Get error message
print("Test 2: Getting auth error message")
msg = auth_manager.get_auth_error_message("meta-llama/Llama-3.1-70B")
print(msg)

# Test 3: Show status
print("\nTest 3: Showing status")
auth_manager.show_status()

print("\n✅ All tests completed successfully!")
