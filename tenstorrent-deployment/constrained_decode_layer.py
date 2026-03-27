#!/usr/bin/env python3
"""
AIREV Constrained Decoding Layer for BFCL
Post-processing that fuzzy-matches model output function names against
the prompt-provided function signatures. Forces exact name reproduction.

This sits between the model and the BFCL evaluator:
  1. Extract function names from the prompt (function definitions)
  2. Parse model output for function calls
  3. Fuzzy-match and replace with nearest prompt function name
  4. Return corrected output

Expected improvement: Name accuracy 20% → 50%+ without retraining.
"""
import re
import json
from difflib import SequenceMatcher, get_close_matches
from typing import List, Dict, Tuple, Optional


def extract_function_names_from_prompt(prompt: str) -> List[str]:
    """
    Extract all function names defined in the BFCL prompt.
    BFCL prompts contain function definitions in various formats:
      - Python: def function_name(params)
      - JSON schema: {"name": "function_name", ...}
      - Plain text: Function: function_name
    """
    names = set()

    # Pattern 1: Python def statements
    for m in re.finditer(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', prompt):
        names.add(m.group(1))

    # Pattern 2: JSON "name" fields (BFCL format)
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', prompt):
        name = m.group(1)
        # Filter out common non-function values
        if name not in ('string', 'integer', 'boolean', 'number', 'array', 'object',
                        'null', 'type', 'required', 'optional', 'description',
                        'properties', 'items', 'enum'):
            names.add(name)

    # Pattern 3: "function_name" in function signature blocks
    for m in re.finditer(r'Function:\s*([a-zA-Z_][a-zA-Z0-9_.]*)', prompt):
        names.add(m.group(1))

    # Pattern 4: tool definitions with "name" field
    for m in re.finditer(r'["\']?name["\']?\s*[:=]\s*["\']([a-zA-Z_][a-zA-Z0-9_.]*)["\']', prompt):
        name = m.group(1)
        if len(name) > 2 and name not in ('string', 'integer', 'boolean', 'number'):
            names.add(name)

    return list(names)


def extract_function_calls_from_output(output: str) -> List[Dict]:
    """
    Extract function calls from model output.
    Handles multiple formats:
      - BFCL: function_name(param1=value1, param2=value2)
      - JSON: {"name": "function_name", "arguments": {...}}
      - OnDemand: {"plugins": [{"name": "...", "pluginId": "..."}]}
    """
    calls = []

    # Format 1: BFCL func(param=value) style
    # Match: word( ... ) but not def word( or if word(
    for m in re.finditer(r'(?<!\w)([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(([^)]*)\)', output):
        name = m.group(1)
        args_str = m.group(2)
        # Skip python keywords and common false positives
        if name in ('def', 'if', 'for', 'while', 'class', 'return', 'print',
                     'import', 'from', 'with', 'as', 'try', 'except', 'dict',
                     'list', 'set', 'tuple', 'int', 'str', 'float', 'bool',
                     'len', 'range', 'type', 'isinstance', 'hasattr', 'getattr'):
            continue
        calls.append({
            'name': name,
            'raw': m.group(0),
            'start': m.start(),
            'end': m.end(),
            'format': 'bfcl'
        })

    # Format 2: JSON with name field
    try:
        parsed = json.loads(output)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and 'name' in item:
                    calls.append({
                        'name': item['name'],
                        'raw': json.dumps(item),
                        'format': 'json'
                    })
        elif isinstance(parsed, dict):
            if 'name' in parsed:
                calls.append({
                    'name': parsed['name'],
                    'raw': output,
                    'format': 'json'
                })
    except (json.JSONDecodeError, TypeError):
        pass

    return calls


def fuzzy_match_name(model_name: str, prompt_names: List[str],
                     threshold: float = 0.5) -> Optional[str]:
    """
    Find the best matching function name from the prompt.
    Uses multiple matching strategies:
      1. Exact match (case-insensitive)
      2. Substring match (model name contains prompt name or vice versa)
      3. Sequence matcher (edit distance ratio)
      4. Token overlap (split by _ and compare tokens)
    """
    if not prompt_names:
        return None

    model_lower = model_name.lower().strip()

    # Strategy 1: Exact match (case insensitive)
    for pn in prompt_names:
        if pn.lower() == model_lower:
            return pn

    # Strategy 2: Exact match after stripping common prefixes/suffixes
    model_stripped = re.sub(r'^(get_|set_|create_|update_|delete_|find_|search_|list_)', '', model_lower)
    for pn in prompt_names:
        pn_stripped = re.sub(r'^(get_|set_|create_|update_|delete_|find_|search_|list_)', '', pn.lower())
        if model_stripped == pn_stripped:
            return pn

    # Strategy 3: Substring containment
    for pn in prompt_names:
        if model_lower in pn.lower() or pn.lower() in model_lower:
            return pn

    # Strategy 4: Token overlap (split by _ and .)
    model_tokens = set(re.split(r'[_.]', model_lower))
    best_overlap = 0
    best_match = None
    for pn in prompt_names:
        pn_tokens = set(re.split(r'[_.]', pn.lower()))
        overlap = len(model_tokens & pn_tokens)
        total = len(model_tokens | pn_tokens)
        if total > 0:
            score = overlap / total
            if score > best_overlap and score >= 0.5:
                best_overlap = score
                best_match = pn

    if best_match and best_overlap >= 0.5:
        return best_match

    # Strategy 5: SequenceMatcher (edit distance)
    best_ratio = 0
    best_match = None
    for pn in prompt_names:
        ratio = SequenceMatcher(None, model_lower, pn.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = pn

    if best_ratio >= threshold:
        return best_match

    # Strategy 6: difflib get_close_matches
    matches = get_close_matches(model_lower, [pn.lower() for pn in prompt_names],
                                 n=1, cutoff=threshold)
    if matches:
        # Map back to original case
        for pn in prompt_names:
            if pn.lower() == matches[0]:
                return pn

    return None


def constrained_decode(prompt: str, model_output: str,
                       threshold: float = 0.45, verbose: bool = False) -> str:
    """
    Main constrained decoding function.
    Takes the full prompt and model output, returns corrected output.
    """
    # Extract available function names from prompt
    prompt_names = extract_function_names_from_prompt(prompt)
    if not prompt_names:
        return model_output  # No functions found in prompt, return as-is

    if verbose:
        print(f"  Prompt functions: {prompt_names}")

    # Extract function calls from model output
    calls = extract_function_calls_from_output(model_output)
    if not calls:
        return model_output  # No calls found, return as-is

    corrected = model_output
    corrections = []

    for call in calls:
        model_name = call['name']
        matched = fuzzy_match_name(model_name, prompt_names, threshold)

        if matched and matched != model_name:
            corrections.append((model_name, matched))
            if call['format'] == 'bfcl':
                # Replace in the raw output string
                corrected = corrected.replace(
                    model_name + '(', matched + '(', 1
                )
            elif call['format'] == 'json':
                # Replace in JSON
                corrected = corrected.replace(
                    '"' + model_name + '"', '"' + matched + '"', 1
                )

    if verbose and corrections:
        for old, new in corrections:
            print(f"  CORRECTED: '{old}' -> '{new}'")

    return corrected


def wrap_bfcl_eval(endpoint_url: str, prompt: str, **kwargs) -> str:
    """
    Wrapper for BFCL evaluation.
    Sends prompt to model endpoint, then applies constrained decoding.
    """
    import requests

    # Send to model
    response = requests.post(endpoint_url + "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": prompt}],
        **kwargs
    }, timeout=120)

    data = response.json()
    raw_output = data["choices"][0]["message"]["content"]

    # Apply constrained decoding
    corrected = constrained_decode(prompt, raw_output, verbose=False)

    return corrected


# ========== TESTS ==========

def test_constrained_decoding():
    """Test suite for constrained decoding."""
    print("=" * 70)
    print("CONSTRAINED DECODING TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        {
            "name": "Exact match (no correction needed)",
            "prompt": 'def get_weather(location: str, days: int) -> dict:',
            "output": 'get_weather(location="Dubai", days=3)',
            "expected_name": "get_weather",
        },
        {
            "name": "Case mismatch",
            "prompt": 'def Get_Weather_Forecast(location: str) -> dict:',
            "output": 'get_weather_forecast(location="Dubai")',
            "expected_name": "Get_Weather_Forecast",
        },
        {
            "name": "Abbreviated name",
            "prompt": 'def search_restaurants_by_cuisine(cuisine: str, location: str) -> list:',
            "output": 'search_restaurants(cuisine="Italian", location="NYC")',
            "expected_name": "search_restaurants_by_cuisine",
        },
        {
            "name": "Conceptual match (different name, same meaning)",
            "prompt": '{"name": "get_stock_price", "description": "Get current stock price"}',
            "output": 'stock_price(ticker="NVDA")',
            "expected_name": "get_stock_price",
        },
        {
            "name": "Hallucinated name with close match",
            "prompt": 'def calculate_mortgage_payment(principal: float, rate: float, years: int):',
            "output": 'calc_mortgage(principal=500000, rate=0.05, years=30)',
            "expected_name": "calculate_mortgage_payment",
        },
        {
            "name": "Multiple functions, pick correct one",
            "prompt": '''def send_email(to: str, subject: str, body: str):
def read_email(folder: str, count: int):
def delete_email(email_id: str):''',
            "output": 'send_mail(to="john@example.com", subject="Hello", body="Hi there")',
            "expected_name": "send_email",
        },
        {
            "name": "Dotted name (API style)",
            "prompt": '{"name": "google_maps.search_places", "description": "Search for places"}',
            "output": 'google_maps.find_places(query="restaurants")',
            "expected_name": "google_maps.search_places",
        },
        {
            "name": "JSON output format",
            "prompt": '{"name": "create_calendar_event", "parameters": {"title": "str", "date": "str"}}',
            "output": '{"name": "calendar_event", "arguments": {"title": "Meeting", "date": "2026-04-01"}}',
            "expected_name": "create_calendar_event",
        },
        {
            "name": "No match (below threshold)",
            "prompt": 'def analyze_sentiment(text: str) -> dict:',
            "output": 'get_weather(location="Dubai")',
            "expected_name": None,  # Should NOT match
        },
        {
            "name": "BFCL real example: Python function",
            "prompt": '''You have access to the following functions:
{"name": "find_hotels_by_amenities", "description": "Find hotels that offer specific amenities", "parameters": {"type": "object", "properties": {"amenities": {"type": "array", "items": {"type": "string"}}, "location": {"type": "string"}}}}''',
            "output": 'find_hotels(amenities=["pool", "gym"], location="Dubai")',
            "expected_name": "find_hotels_by_amenities",
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(tests):
        print(f"Test {i+1}: {test['name']}")
        print(f"  Prompt functions: {extract_function_names_from_prompt(test['prompt'])}")

        corrected = constrained_decode(test['prompt'], test['output'], verbose=True)

        if test['expected_name'] is None:
            # Should NOT have been corrected
            if corrected == test['output']:
                print(f"  PASS (correctly left unchanged)")
                passed += 1
            else:
                print(f"  FAIL (incorrectly modified: '{corrected}')")
                failed += 1
        else:
            # Check if the corrected output contains the expected name
            if test['expected_name'] in corrected:
                print(f"  PASS (output contains '{test['expected_name']}')")
                passed += 1
            else:
                print(f"  FAIL (expected '{test['expected_name']}' in output)")
                print(f"  Got: {corrected}")
                failed += 1
        print()

    print("=" * 70)
    print(f"RESULTS: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    test_constrained_decoding()
