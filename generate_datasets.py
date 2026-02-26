#!/usr/bin/env python3
import os
import json
import re
import sys
import subprocess
import importlib.util
from typing import Optional, Tuple, Dict, Any

# Try to import OpenAI SDK
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai", file=sys.stderr)

# -----------------------------
# Configuration - OpenAI
# -----------------------------
# Set API key if not in environment (you can set it here for testing)
if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_KEY"):
    ### Enter you key here
    os.environ["OPENAI_API_KEY"] = ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # gpt-5, gpt-5-mini, gpt-5-nano, etc.

# Output token budget
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "8000"))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")  # low/medium/high
ENABLE_REASONING_SUMMARY = os.getenv("ENABLE_REASONING_SUMMARY", "false").lower() == "true"

# Text truncation limits (characters)
DETECTION_TEXT_LIMIT = int(os.getenv("DETECTION_TEXT_LIMIT", "50000"))
GENERATION_TEXT_LIMIT = int(os.getenv("GENERATION_TEXT_LIMIT", "200000"))

# -----------------------------
# OpenAI client
# -----------------------------
_openai_client = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY (or OPENAI_KEY) not set. Please set it as an environment variable.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def call_openai_api(
    prompt: str,
    model: Optional[str] = None,
    response_format: Optional[dict] = None,
    thinking: bool = False,
) -> Optional[dict]:
    """
    Calls OpenAI.
    - If response_format is provided, uses Chat Completions (supports json_schema).
    - Otherwise, uses Responses API (supports reasoning config).
    Returns: {"content": str, "thinking": str}
    """
    if model is None:
        model = OPENAI_MODEL

    if not OPENAI_AVAILABLE:
        print("Error: OpenAI library not installed.", file=sys.stderr)
        return None

    client = get_openai_client()
    is_reasoning_model = any(x in model.lower() for x in ["gpt-5", "o1", "o4"])

    try:
        # Structured outputs path
        if response_format is not None:
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": response_format,
            }

            # gpt-5 uses max_completion_tokens
            if "gpt-5" in model.lower():
                params["max_completion_tokens"] = MAX_OUTPUT_TOKENS
            else:
                params["max_tokens"] = MAX_OUTPUT_TOKENS
                params["temperature"] = 0.2

            resp = client.chat.completions.create(**params)
            content = resp.choices[0].message.content or ""
            return {"content": content, "thinking": ""}

        # Non-structured path (Responses API)
        params = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }

        if is_reasoning_model and thinking:
            reasoning_cfg = {"effort": REASONING_EFFORT}
            if ENABLE_REASONING_SUMMARY:
                reasoning_cfg["summary"] = "auto"
            params["reasoning"] = reasoning_cfg

        resp = client.responses.create(**params)
        content = getattr(resp, "output_text", "") or ""

        reasoning_trace = ""
        if hasattr(resp, "output") and isinstance(resp.output, list):
            for item in resp.output:
                if isinstance(item, dict) and item.get("type") == "reasoning" and "summary" in item:
                    for s in item.get("summary", []):
                        if s.get("type") == "summary_text":
                            reasoning_trace += s.get("text", "") + "\n"

        return {"content": content, "thinking": reasoning_trace.strip()}

    except Exception as e:
        print(f"Error calling OpenAI API: {e}", file=sys.stderr)
        return None


def call_llm(prompt: str, response_format: Optional[dict] = None, thinking: bool = False) -> Optional[dict]:
    return call_openai_api(prompt, response_format=response_format, thinking=thinking)


# -----------------------------
# File reading
# -----------------------------
def read_pages(book_dir: str, start_page: int, num_pages: int = 3) -> Optional[str]:
    pages_text = []
    for i in range(start_page, start_page + num_pages):
        page_file = os.path.join(book_dir, f"{i}.txt")
        if not os.path.exists(page_file):
            return None
        with open(page_file, "r", encoding="utf-8") as f:
            pages_text.append(f.read())
    return "\n\n".join(pages_text)


# -----------------------------
# Category detection (now includes tool_use)
# -----------------------------
def detect_content_categories(text: str, model: Optional[str] = None) -> dict:
    if model is None:
        model = OPENAI_MODEL

    text_sample = text[:DETECTION_TEXT_LIMIT] if len(text) > DETECTION_TEXT_LIMIT else text

    prompt = f"""Analyze the following text from a quantitative finance or mathematics book.
Categorize the content into one or more categories:

1) instruction_tuning: concepts, explanations, intuition, definitions, Q&A
2) reasoning: derivations, proofs, multi-step problem solving, worked examples
3) tool_use: explicit formulas/algorithms suitable for implementing a function and computing numeric outputs

Text:
---
{text_sample}
---

Respond ONLY with JSON:
{{
  "instruction_tuning": true/false,
  "reasoning": true/false,
  "tool_use": true/false
}}"""

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "content_categories",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "instruction_tuning": {"type": "boolean"},
                    "reasoning": {"type": "boolean"},
                    "tool_use": {"type": "boolean"},
                },
                "required": ["instruction_tuning", "reasoning", "tool_use"],
                "additionalProperties": False,
            },
        },
    }

    result = call_llm(prompt, response_format=response_format, thinking=False)
    if not result:
        return {"instruction_tuning": True, "reasoning": False, "tool_use": False}

    try:
        return json.loads(result["content"])
    except Exception:
        return {"instruction_tuning": True, "reasoning": False, "tool_use": False}


# -----------------------------
# Schemas
# -----------------------------
def get_instruction_tuning_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["user", "assistant"]},
                        "content": {"type": "string"},
                    },
                    "required": ["role", "content"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["messages"],
        "additionalProperties": False,
    }


def get_tool_use_dataset_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "tool"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["assistant"]},
                                "content": {"type": "string"},
                                "tool_calls": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "arguments": {"type": "string"},
                                        },
                                        "required": ["name", "arguments"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["role", "content", "tool_calls"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["assistant"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                            "additionalProperties": False,
                        },
                    ],
                },
            },
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["function"]},
                        "function": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["object"]},
                                        "properties": {
                                            "type": "object",
                                            "properties": {
                                                "code": {
                                                    "type": "object",
                                                    "properties": {
                                                        "type": {"type": "string", "enum": ["string"]},
                                                    },
                                                    "required": ["type"],
                                                    "additionalProperties": False,
                                                },
                                                "inputs": {
                                                    "type": "object",
                                                    "properties": {
                                                        "type": {"type": "string", "enum": ["object"]},
                                                    },
                                                    "required": ["type"],
                                                    "additionalProperties": False,
                                                },
                                            },
                                            "required": ["code", "inputs"],
                                            "additionalProperties": False,
                                        },
                                        "required": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "additionalProperties": {
                                            "type": "boolean",
                                            "enum": [False],
                                        },
                                    },
                                    "required": ["type", "properties", "required", "additionalProperties"],
                                    "additionalProperties": False,
                                },
                            },
                            "required": ["name", "description", "parameters"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["type", "function"],
                    "additionalProperties": False,
                },
            },
            "success": {"type": "string", "enum": ["yes", "no"]},
            "ground_truth": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["value", "description"],
                "additionalProperties": False,
            },
        },
        "required": ["messages", "tools", "success", "ground_truth"],
        "additionalProperties": False,
    }


# -----------------------------
# Tool execution and testing
# -----------------------------
def extract_code_and_inputs(tool_call_args: str) -> Tuple[Optional[str], Optional[dict]]:
    """Extract Python code and inputs from tool call arguments JSON string."""
    try:
        args = json.loads(tool_call_args)
        code = args.get("code", "")
        inputs = args.get("inputs", {})
        # If inputs is None or empty, try to get other parameters that might be inputs
        if not inputs:
            # Look for common input parameter names
            input_keys = ["S", "K", "T", "r", "sigma", "q", "returns", "risk_free_rate", "volatility"]
            inputs = {k: v for k, v in args.items() if k != "code" and k in input_keys}
            # If still empty, use all non-code keys as inputs
            if not inputs:
                inputs = {k: v for k, v in args.items() if k != "code"}
        return code, inputs if inputs else {}
    except Exception as e:
        print(f"Error extracting code/inputs: {e}", file=sys.stderr)
        return None, None


def detect_required_imports(code: str) -> list:
    """Detect required imports from code."""
    imports = []
    # Available packages: scipy, matplotlib, numpy, pandas, scikit-learn, statsmodels
    if "import numpy" in code or "import np" in code or "from numpy" in code:
        imports.append("numpy")
    if "import scipy" in code or "from scipy" in code:
        imports.append("scipy")
    if "import pandas" in code or "from pandas" in code or "import pd" in code:
        imports.append("pandas")
    if "import matplotlib" in code or "from matplotlib" in code or "import plt" in code:
        imports.append("matplotlib")
    if "import sklearn" in code or "from sklearn" in code or "import scikit-learn" in code:
        imports.append("scikit-learn")
    if "import statsmodels" in code or "from statsmodels" in code:
        imports.append("statsmodels")
    if "import math" in code or "from math" in code:
        # math is built-in, no need to install
        pass
    return imports


def install_package(package: str) -> bool:
    """Install a Python package using pip."""
    try:
        print(f"  Installing {package}...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        print(f"  Failed to install {package}: {e}", file=sys.stderr)
        return False


def execute_python_code(code: str, inputs: dict) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Execute Python code with given inputs.
    Returns: (success: bool, result: Any, error: str or None)
    """
    # Create a safe execution environment
    exec_globals = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "__file__": None,
    }
    exec_locals = {}
    
    try:
        # Add inputs to execution environment
        exec_locals.update(inputs)
        
        # Execute the code
        exec(code, exec_globals, exec_locals)
        
        # Try to find a return value or result
        # Look for common result variable names
        result = None
        for var_name in ["result", "output", "value", "answer"]:
            if var_name in exec_locals:
                result = exec_locals[var_name]
                break
        
        # If code defines a function, try to call it with inputs
        if not result:
            # Look for function definitions
            for key, value in exec_locals.items():
                if callable(value) and not key.startswith("_"):
                    try:
                        # Try calling with inputs as keyword arguments
                        result = value(**inputs)
                        break
                    except Exception:
                        try:
                            # Try calling with inputs as positional arguments
                            if inputs:
                                result = value(*inputs.values())
                                break
                        except Exception:
                            pass
        
        return True, result, None
    except Exception as e:
        return False, None, str(e)


def create_test_script(code: str, inputs: dict, test_id: str, output_dir: str = "tests") -> str:
    """Create a test script file in tests/ folder."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize test_id for filename
    safe_test_id = re.sub(r'[^\w\-_]', '_', test_id)
    test_file = os.path.join(output_dir, f"test_{safe_test_id}.py")
    
    script_content = f'''#!/usr/bin/env python3
"""
Test script for tool-use dataset: {test_id}
Generated automatically by generate_datasets.py
"""

import json

# Inputs
inputs = {json.dumps(inputs, indent=2)}

# Code to test
code = """{code}"""

# Execute
exec_globals = {{"__builtins__": __builtins__, "__name__": "__main__"}}
exec_locals = {{}}
exec_locals.update(inputs)

try:
    exec(code, exec_globals, exec_locals)
    
    # First, try to find and call a function (preferred approach)
    result = None
    function_found = False
    
    # Look for function definitions (excluding built-ins and private functions)
    for key, value in exec_locals.items():
        if callable(value) and not key.startswith("_") and key not in ["print", "json", "exit"]:
            try:
                # Try calling with inputs as keyword arguments
                result = value(**inputs)
                function_found = True
                break
            except Exception:
                try:
                    # Try calling with inputs as positional arguments
                    if inputs:
                        result = value(*inputs.values())
                        function_found = True
                        break
                except Exception:
                    pass
    
    # If no function found, look for result variables (fallback)
    if not function_found:
        for var_name in ["result", "output", "value", "answer"]:
            if var_name in exec_locals:
                result = exec_locals[var_name]
                break
    
    print(f"SUCCESS: {{result}}")
    print(json.dumps({{"success": True, "result": result}}))
except Exception as e:
    print(f"FAILED: {{e}}")
    print(json.dumps({{"success": False, "error": str(e)}}))
    exit(1)
'''
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    return test_file


def test_tool_code(dataset: dict, test_id: str) -> Tuple[bool, str]:
    """
    Test the tool code from a dataset.
    Returns: (success: bool, test_file_path: str)
    """
    # Find the tool call in messages
    tool_call = None
    for msg in dataset.get("messages", []):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg.get("tool_calls", []):
                if tc.get("name") == "evaluate_quant_function":
                    tool_call = tc
                    break
            if tool_call:
                break
    
    if not tool_call:
        return False, ""
    
    # Extract code and inputs
    code, inputs = extract_code_and_inputs(tool_call.get("arguments", "{}"))
    if not code:
        print(f"  No code found in tool call", file=sys.stderr)
        return False, ""
    
    if not inputs:
        print(f"  Warning: No inputs found in tool call", file=sys.stderr)
    
    # Detect and install required packages
    required_imports = detect_required_imports(code)
    for pkg in required_imports:
        # Check if package is already installed
        try:
            importlib.import_module(pkg)
        except ImportError:
            if not install_package(pkg):
                print(f"  Warning: Could not install {pkg}, test may fail", file=sys.stderr)
    
    # Create test script
    test_file = create_test_script(code, inputs or {}, test_id)
    
    # Execute the test
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, test_file
        else:
            if result.stderr:
                print(f"  Test error: {result.stderr[:200]}", file=sys.stderr)
            return False, test_file
    except subprocess.TimeoutExpired:
        print(f"  Test timed out after 30 seconds", file=sys.stderr)
        return False, test_file
    except Exception as e:
        print(f"Error running test: {e}", file=sys.stderr)
        return False, test_file


# -----------------------------
# Dataset generators
# -----------------------------
def generate_instruction_tuning_dataset(pages_text: str, book_name: str, page_range: str, existing_questions: list = None) -> Optional[dict]:
    """
    Generate ONE question-answer pair from the text.
    If existing_questions is provided, avoid topics already covered.
    """
    if existing_questions is None:
        existing_questions = []
    
    existing_context = ""
    if existing_questions:
        existing_context = "\n\nIMPORTANT: The following questions have already been extracted from this text. Please find a DIFFERENT concept/topic:\n"
        for i, qa in enumerate(existing_questions, 1):
            existing_context += f"{i}. {qa.get('user_question', '')}\n"
        existing_context += "\nExtract a NEW question about a DIFFERENT concept from the text.\n"
    
    prompt = f"""You are an expert in quantitative finance.
Extract and create ONE question-answer pair based on the ACTUAL content from the text.
{existing_context}
IMPORTANT: The text already contains explanations, definitions, formulas, or worked examples. Your task is to:
1. Identify a key concept, definition, or explanation from the text (different from any already extracted)
2. Create a question that asks about this concept
3. Use the ACTUAL explanation/formula/definition from the text as the answer

Constraints:
- Base the question on what's actually in the text, don't invent new concepts
- Use the exact formulas, definitions, or explanations from the text when possible
- No calculations or tool use needed
- The answer should reflect what the text teaches
- Choose a DIFFERENT topic/concept than any already extracted

Text ("{book_name}", pages {page_range}):
---
{pages_text[:GENERATION_TEXT_LIMIT]}
---

Return JSON:
{{
  "messages": [
    {{"role":"user","content":"Question based on the text content..."}},
    {{"role":"assistant","content":"Answer using the actual explanation/formula from the text..."}}
  ]
}}"""

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "instruction_tuning_dataset",
            "strict": True,
            "schema": get_instruction_tuning_schema(),
        },
    }

    result = call_llm(prompt, response_format=response_format, thinking=False)
    if not result:
        return None

    try:
        dataset = json.loads(result["content"])
        # Extract the question for tracking
        if dataset.get("messages") and len(dataset["messages"]) >= 2:
            user_msg = dataset["messages"][0]
            if user_msg.get("role") == "user":
                dataset["_user_question"] = user_msg.get("content", "")
        return dataset
    except Exception:
        m = re.search(r"\{.*\}", result.get("content", ""), re.DOTALL)
        if m:
            try:
                dataset = json.loads(m.group(0))
                if dataset.get("messages") and len(dataset["messages"]) >= 2:
                    user_msg = dataset["messages"][0]
                    if user_msg.get("role") == "user":
                        dataset["_user_question"] = user_msg.get("content", "")
                return dataset
            except Exception:
                return None
        return None


def generate_multiple_instruction_tuning_datasets(pages_text: str, book_name: str, page_range: str, num_iterations: int = 3) -> list:
    """
    Generate multiple question-answer pairs from the same text by running recursively.
    Returns a list of separate datasets, one for each Q&A pair.
    """
    datasets = []
    existing_questions = []
    
    for i in range(num_iterations):
        print(f"    -> Extracting question {i+1}/{num_iterations}...", flush=True)
        dataset = generate_instruction_tuning_dataset(pages_text, book_name, page_range, existing_questions)
        
        if not dataset or not dataset.get("messages"):
            print(f"    -> No more questions found after {i} iterations", flush=True)
            break
        
        # Remove the tracking field before saving
        if "_user_question" in dataset:
            del dataset["_user_question"]
        
        # Add to list of separate datasets
        datasets.append(dataset)
        
        # Track the question for next iteration
        if dataset.get("messages") and len(dataset["messages"]) >= 2:
            user_msg = dataset["messages"][0]
            if user_msg.get("role") == "user":
                existing_questions.append({"user_question": user_msg.get("content", "")})
    
    return datasets


def generate_reasoning_tool_dataset(pages_text: str, book_name: str, page_range: str) -> Optional[dict]:
    """
    Tool-use example:
      - user asks for numeric computation
      - assistant defines python function in tool call ("code") and provides inputs
      - tool returns numeric result
      - assistant interprets
    numpy/scipy ARE allowed.
    """
    prompt = f"""You are creating ONE training example for a quant-finance reasoning + tool-use dataset.

IMPORTANT: The text contains formulas, worked examples, or computational methods. Your task is to:
1. Extract an ACTUAL formula or computational method from the text
2. Find a worked example or calculation in the text that shows the expected result
3. Create a question that asks to compute something using that formula
4. Implement the EXACT formula from the text in Python code
5. Use the SAME input values from the worked example in the text (if available)
6. Extract the GROUND TRUTH (expected answer) from the text's worked example/calculation

Hard constraints:
- Use the ACTUAL formula/method from the text, don't invent new ones
- The ground truth value should come from a worked example or calculation shown in the text
- If the text shows a calculation with specific inputs and result, use those exact values
- The tool call must include Python code in a string under key "code"
- Available packages: numpy, scipy, pandas, matplotlib, scikit-learn, statsmodels
- Code must be a CLEAN Python function definition that takes inputs as parameters and returns a numeric scalar
- DO NOT include function calls or result assignments in the code - just the function definition
- Example format: "def function_name(param1, param2):\\n    # implementation\\n    return result"
- Provide specific numeric inputs (preferably from the worked example in the text)
- Tool response must contain one numeric output: {{"result": {{"value": ... }}}}
- Final assistant message must state the result and interpret it briefly
- Ground truth must be the actual numeric result shown in the text's worked example

Text ("{book_name}", pages {page_range}):
---
{pages_text[:GENERATION_TEXT_LIMIT]}
---

Return JSON EXACTLY:
{{
  "messages": [
    {{
      "role": "user",
      "content": "Compute ... with given numbers ..."
    }},
    {{
      "role": "assistant",
      "content": "I will define the function and evaluate it.",
      "tool_calls": [
        {{
          "name": "evaluate_quant_function",
          "arguments": "{{\\"code\\": \\"...python...\\", \\"inputs\\": {{...}}}}"
        }}
      ]
    }},
    {{
      "role": "tool",
      "content": "{{\\"result\\": {{\\"value\\": 0.0}}}}"
    }},
    {{
      "role": "assistant",
      "content": "State the numeric result and give a brief interpretation."
    }}
  ],
  "tools": [
    {{
      "type": "function",
      "function": {{
        "name": "evaluate_quant_function",
        "description": "Executes provided Python code that computes a quantitative finance metric from inputs.",
        "parameters": {{
          "type": "object",
          "properties": {{
            "code": {{"type": "string"}},
            "inputs": {{"type": "object"}}
          }},
          "required": ["code", "inputs"],
          "additionalProperties": false
        }}
      }}
    }}
  ],
  "ground_truth": {{
    "value": <extract the actual numeric result from the worked example in the text>,
    "description": "Expected numeric result from the worked example/calculation shown in the text"
  }}
}}"""

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "reasoning_tool_dataset",
            "strict": True,
            "schema": get_tool_use_dataset_schema(),
        },
    }

    result = call_llm(prompt, response_format=response_format, thinking=False)
    if not result:
        return None

    try:
        dataset = json.loads(result["content"])
    except Exception:
        m = re.search(r"\{.*\}", result.get("content", ""), re.DOTALL)
        if m:
            try:
                dataset = json.loads(m.group(0))
            except Exception:
                return None
        else:
            return None
    
    # Extract ground truth from the dataset (should be provided by model from text)
    # If not present or is 0.0, try to compute it as fallback
    if "ground_truth" not in dataset or dataset.get("ground_truth", {}).get("value") == 0.0:
        # Try to compute ground truth by executing the code
        ground_truth = compute_ground_truth(dataset)
        if ground_truth:
            dataset["ground_truth"] = ground_truth
        else:
            # Fallback: use value from tool response if available
            tool_value = extract_tool_result_value(dataset)
            dataset["ground_truth"] = {
                "value": tool_value if tool_value is not None else 0.0,
                "description": "Expected numeric result (computed from code execution, ground truth not found in text)"
            }
    else:
        # Ground truth was extracted from text by the model
        if "description" not in dataset.get("ground_truth", {}):
            dataset["ground_truth"]["description"] = "Expected numeric result extracted from worked example in the text"
    
    # Add success field (will be set after testing)
    dataset["success"] = "no"
    return dataset


def extract_tool_result_value(dataset: dict) -> Optional[float]:
    """Extract the numeric value from the tool response message."""
    for msg in dataset.get("messages", []):
        if msg.get("role") == "tool":
            try:
                content = json.loads(msg.get("content", "{}"))
                result = content.get("result", {})
                value = result.get("value")
                if isinstance(value, (int, float)):
                    return float(value)
            except Exception:
                pass
    return None


def compute_ground_truth(dataset: dict) -> Optional[dict]:
    """
    Compute ground truth by executing the code from the tool call.
    Returns: {"value": float, "description": str} or None if computation fails
    """
    # Find the tool call in messages
    tool_call = None
    for msg in dataset.get("messages", []):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg.get("tool_calls", []):
                if tc.get("name") == "evaluate_quant_function":
                    tool_call = tc
                    break
            if tool_call:
                break
    
    if not tool_call:
        return None
    
    # Extract code and inputs
    code, inputs = extract_code_and_inputs(tool_call.get("arguments", "{}"))
    if not code or not inputs:
        return None
    
    # Try to execute and get the result
    success, result, error = execute_python_code(code, inputs)
    
    if success and result is not None:
        try:
            # Ensure result is numeric
            numeric_value = float(result)
            return {
                "value": numeric_value,
                "description": f"Expected result from executing the formula with inputs: {inputs}"
            }
        except (ValueError, TypeError):
            return None
    
    return None


# -----------------------------
# Processing logic: make exactly 2 outputs total
# -----------------------------
GOT_KNOWLEDGE = False
GOT_REASONING_TOOL = False


def process_books(book_txt_dir: str, output_dir: str):
    global GOT_KNOWLEDGE, GOT_REASONING_TOOL

    # Find all book directories
    book_dirs = []
    for entry in os.listdir(book_txt_dir):
        p = os.path.join(book_txt_dir, entry)
        if os.path.isdir(p):
            book_dirs.append(p)

    if not book_dirs:
        print(f"No book directories found in '{book_txt_dir}'.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    knowledge_out = os.path.join(output_dir, "knowledge.json")
    reasoning_out = os.path.join(output_dir, "reasoning_tool.json")

    print(f"Found {len(book_dirs)} book(s) to scan")

    for book_dir in book_dirs:
        if GOT_KNOWLEDGE and GOT_REASONING_TOOL:
            break

        book_name = os.path.basename(book_dir)
        print(f"\nScanning book: {book_name}")

        # Determine page count (1..N)
        pages = []
        for i in range(1, 10000):
            page_file = os.path.join(book_dir, f"{i}.txt")
            if os.path.exists(page_file):
                pages.append(i)
            else:
                break

        if not pages:
            print(f"  No pages found in {book_dir}")
            continue

        total_pages = len(pages)
        print(f"  Found {total_pages} pages")

        # Sliding 3-page windows
        for start_page in range(1, total_pages - 1):
            if GOT_KNOWLEDGE and GOT_REASONING_TOOL:
                break

            page_range = f"{start_page}-{start_page+2}"
            pages_text = read_pages(book_dir, start_page, 3)
            if not pages_text:
                continue

            print(f"  [p{start_page}/{total_pages}] Categorizing pages {page_range} ...", flush=True)
            cats = detect_content_categories(pages_text)

            instruction_tuning = bool(cats.get("instruction_tuning"))
            reasoning = bool(cats.get("reasoning"))
            tool_use = bool(cats.get("tool_use"))

            print(f"    categories: instruction_tuning={instruction_tuning}, reasoning={reasoning}, tool_use={tool_use}", flush=True)

            # 1) Generate multiple knowledge datasets (one file per Q&A pair)
            if instruction_tuning and not GOT_KNOWLEDGE:
                print("    -> Generating KNOWLEDGE datasets (extracting multiple Q&A pairs)...", flush=True)
                datasets = generate_multiple_instruction_tuning_datasets(pages_text, book_name, page_range, num_iterations=3)
                if datasets:
                    # Save each Q&A pair as a separate JSON file
                    for idx, ds in enumerate(datasets, 1):
                        output_file = os.path.join(output_dir, f"knowledge_p{start_page}_{idx}.json")
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(ds, f, indent=2, ensure_ascii=False)
                        print(f"    ✓ wrote {output_file}", flush=True)
                    GOT_KNOWLEDGE = True
                    print(f"    ✓ Generated {len(datasets)} knowledge dataset(s)")

            # 2) ONE reasoning/tool dataset total (triggered by tool_use OR reasoning)
            if (tool_use or reasoning) and not GOT_REASONING_TOOL:
                print("    -> Generating REASONING/TOOL dataset ...", flush=True)
                ds = generate_reasoning_tool_dataset(pages_text, book_name, page_range)
                if ds:
                    # Test the tool code and mark success
                    test_id = f"{book_name.replace(' ', '_')}_p{start_page}"
                    success, test_file = test_tool_code(ds, test_id)
                    ds["success"] = "yes" if success else "no"
                    
                    if success:
                        print(f"    ✓ Test passed: {test_file}", flush=True)
                    else:
                        print(f"    ✗ Test failed: {test_file}", flush=True)
                    
                    with open(reasoning_out, "w", encoding="utf-8") as f:
                        json.dump(ds, f, indent=2, ensure_ascii=False)
                    GOT_REASONING_TOOL = True
                    print(f"    ✓ wrote {reasoning_out} (success: {ds['success']})")

            del pages_text

    print("\nDone.")
    print(f"Knowledge dataset: {'created' if GOT_KNOWLEDGE else 'NOT created'} -> {knowledge_out}")
    print(f"Reasoning/tool dataset: {'created' if GOT_REASONING_TOOL else 'NOT created'} -> {reasoning_out}")

    if not GOT_KNOWLEDGE or not GOT_REASONING_TOOL:
        print("\nNote: If one file is missing, it means the model didn't detect suitable pages.")
        print("Try increasing DETECTION_TEXT_LIMIT/GENERATION_TEXT_LIMIT or scan different books/pages.")


# -----------------------------
# Main
# -----------------------------
def main():
    book_txt_dir = "book_txt"
    output_dir = "dataset"

    print("\n" + "=" * 60)
    print("OpenAI API Configuration")
    print("=" * 60)
    print(f"Model: {OPENAI_MODEL}")
    print(f"API Key: {'Set' if OPENAI_API_KEY else 'Not Set'}")
    print(f"Max Output Tokens: {MAX_OUTPUT_TOKENS}")
    print(f"Reasoning Effort: {REASONING_EFFORT}")
    print(f"Reasoning Summary: {'Enabled' if ENABLE_REASONING_SUMMARY else 'Disabled'}")
    print(f"Detection Text Limit: {DETECTION_TEXT_LIMIT} chars")
    print(f"Generation Text Limit: {GENERATION_TEXT_LIMIT} chars")
    print("=" * 60 + "\n")

    if not os.path.isdir(book_txt_dir):
        print(f"Directory '{book_txt_dir}' not found. Please run convert_all_pdfs.py first.", file=sys.stderr)
        sys.exit(1)

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not set. Export it, e.g.:", file=sys.stderr)
        print("  export OPENAI_API_KEY='sk-...'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    process_books(book_txt_dir, output_dir)


if __name__ == "__main__":
    main()
