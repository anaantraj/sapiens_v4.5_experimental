import re
import json_repair

def parse_part_b_response(response_text):
    """Parses Part B (Root Cause Analysis) output."""
    if not response_text:
        return "[LLM Error] [Empty response]"

    try:
        # 1. Check for [Motive] [Explanation]
        match = re.search(r"(\[.*?\]\s*\[.*?\])", response_text, re.DOTALL)
        if match:
            analysis = match.group(1).strip()
            # Check for "no missed motives"
            if "no missed motives" in analysis.lower() or "none" in analysis.lower():
                return "[No missed motives] [No additional themes missed]"
            return analysis

        # 2. Check for [Motive] Explanation
        match_flex = re.search(r"(\[.*?\])\s*(.*)", response_text, re.DOTALL)
        if match_flex:
            return f"{match_flex.group(1).strip()} [{match_flex.group(2).strip()}]"

        # 3. JSON fallback
        if response_text.startswith("{"):
            parsed = json_repair.loads(response_text)
            if isinstance(parsed, dict) and "analysis" in parsed:
                return parsed["analysis"]

        return f"[LLM Error] [Format mismatch: {response_text}]"
    except Exception as e:
        return f"[LLM Error] [Parsing failed: {str(e)}]"