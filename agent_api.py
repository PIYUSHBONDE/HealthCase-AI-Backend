from typing import Dict, Any, Tuple
import json
import re

def normalize_agent_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Vertex AI agent response.
    Extracts:
      - thoughts
      - function calls
      - text content
      - parsed testcases (only if response contains JSON testcases)
      Falls back to normal text if parsing fails.
    """

    def _strip_code_fences(text: str) -> str:
        """Remove leading ```json fences and trailing whitespace."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        cleaned = text.strip()
        cleaned = re.sub(r"^\s*?```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned

    def _extract_first_json_object(text: str) -> str:
        """Extract the first top-level {...} JSON object using brace counting."""
        s = _strip_code_fences(text)
        start = s.find("{")
        if start == -1:
            raise ValueError("No JSON object start '{' found in text")
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(s[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
        raise ValueError("Unbalanced braces: could not find the end of the JSON object")

    def parse_testcases_payload(body: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Accepts a dict like {"text": "<prose...>"} and returns:
        (parsed_inner_json_dict, "")
        On failure, returns ({}, "<error_message>")
        """
        if not isinstance(body, dict):
            return {}, "Input body must be a dict"
        raw_text = body.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            return {}, "Body must include a non-empty 'text' string"
        try:
            json_block = _extract_first_json_object(raw_text)
        except Exception as e:
            return {}, f"JSON extract error: {e}"
        try:
            parsed = json.loads(json_block)
        except json.JSONDecodeError as e:
            return {}, f"JSON parse error: {e}"
        return parsed, ""

    # --- Main normalization ---
    normalized = {
        "text": [],
        "thoughts": [],
        "function_calls": [],
        "metadata": {},
        # parsed_testcases will only be added if JSON parsing succeeds
    }

    parts = payload.get("content", {}).get("parts", [])
    for part in parts:
        if "text" in part:
            normalized["text"].append(part["text"])
        if "thought_signature" in part:
            normalized["thoughts"].append(part["thought_signature"])
        if "function_call" in part:
            normalized["function_calls"].append(part["function_call"])

    # Flatten text
    text_response = "\n".join(normalized["text"])
    normalized["text"] = text_response

    # Attempt to parse testcases JSON if it looks like a testcase response
    if "testcase" in text_response.lower() or "test case" in text_response.lower():
        parsed, err = parse_testcases_payload({"text": text_response})
        if parsed and not err:
            normalized["parsed_testcases"] = parsed
        # If parsing fails, fallback to normal text only

    return normalized