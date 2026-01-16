import logging
import json
from typing import AsyncGenerator, Any, Dict
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
import uuid
import json
import logging
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

@retry(
    wait=wait_random_exponential(multiplier=2, max=60),  # Wait 2s, 4s, 8s... up to 60s
    stop=stop_after_attempt(10),  # Retry 10 times before failing
    retry=retry_if_exception_type(ResourceExhausted)  # Only retry on 429 errors
)
async def summarize_testcases_from_markdown(
    current_testcases: str, 
    model_name: str = "gemini-2.0-flash"
) -> str:
    """
    Summarizes test cases from markdown format into a user-friendly message.
    
    Args:
        current_testcases: Markdown string containing either:
            - A test case table with optional compliance rules section, OR
            - An error message explaining why test cases cannot be generated
        model_name: Model identifier (default: gemini-2.0-flash)
        
    Returns:
        Formatted summary message for the user
    """
    
    summarization_prompt = f"""
You are a test case report generator. Analyze the input and create a clear, professional summary message for the user.

**Input:**
{current_testcases}

**Instructions:**

**CASE 1: If the input contains an error message or indicates test cases CANNOT be generated:**
Generate a message that:
1. Clearly states that test cases could not be generated
2. Explains the specific reason/issue mentioned in the input
3. Provides constructive guidance on what the user should do next (e.g., provide more details, clarify requirements, add documentation)
4. Maintains a helpful and professional tone
5. Uses markdown formatting with appropriate emojis

**Example output for error case:**
## ⚠️ Unable to Generate Test Cases

I was unable to generate test cases for your request due to insufficient information in the provided requirements.

### 🔍 Issue Identified:
- The requirements document lacks specific functional details needed for test case generation
- Missing information about expected user flows and system behaviors

### 📝 Recommended Actions:
- Provide a detailed Business Requirements Document (BRD) or Functional Specification
- Include specific user stories with acceptance criteria
- Clarify expected system behaviors and validation rules
- Add information about compliance requirements if applicable

Once you provide additional details, I'll be able to generate comprehensive test cases for your project.

---

**CASE 2: If the input contains a valid test case table:**
Generate a concise, user-friendly summary message that:
1. Starts with a success confirmation
2. Counts and mentions the total number of test cases generated
3. Extracts and lists the main testing areas covered (infer from test descriptions)
4. Highlights any compliance rules mentioned in the "Applied Compliance Rules" section
5. Ends with a positive closing statement

**Example output for success case:**
## ✅ Test Cases Generated Successfully!

I've generated **28 comprehensive test cases** for your healthcare application's user authentication system.

### 📋 Test Coverage Areas:
- User registration and sign-up flow
- Email validation and OTP verification
- Password security and validation rules
- Profile setup and user preferences
- Login authentication (email and biometric)
- Password recovery workflow

### 🔒 Compliance Rules Covered:
Your test cases include validation for **4 critical compliance rules**:
- HIPAA data encryption requirements
- TLS 1.2 secure transmission protocols
- Password hashing and security standards
- PHI audit trail compliance

All test cases are ready for integration with your test management system! 🚀

---

**Formatting Requirements:**
- Use markdown formatting for readability
- Use appropriate section headers (##, ###)
- Use bullet points for lists
- Keep the tone professional yet friendly
- Maximum length: 300 words
- Include relevant emojis for visual appeal

Return ONLY the formatted summary message, no additional explanations or meta-commentary.
"""
    
    try:
        # Initialize Vertex AI Generative Model
        model = GenerativeModel(model_name)
        
        # Generate content
        response = model.generate_content(summarization_prompt)
        summary_message = response.text.strip()
        
        return summary_message
        
    except Exception as e:
        # Fallback to template-based summary if LLM fails
        logger.error(f"Error generating summary with LLM: {e}")
        return generate_fallback_summary_from_markdown(current_testcases)


def generate_fallback_summary_from_markdown(current_testcases: str) -> str:
    """
    Generates a fallback summary message without using LLM.
    Parses markdown string to determine if it's an error or valid test cases.
    
    Args:
        current_testcases: Markdown string with test cases or error message
        
    Returns:
        Formatted summary string
    """
    # Check if it's an error message
    error_indicators = [
        "cannot be generated",
        "unable to generate",
        "insufficient information",
        "missing requirements",
        "insufficient requirements"
    ]
    
    is_error = any(indicator in current_testcases.lower() for indicator in error_indicators)
    
    if is_error:
        # Extract reason if possible
        reason = "Insufficient information provided in the requirements."
        if "due to" in current_testcases.lower():
            try:
                reason = current_testcases.split("due to", 1)[1].strip().rstrip(".")
            except:
                pass
        
        summary_lines = [
            "## ⚠️ Unable to Generate Test Cases\n",
            f"I was unable to generate test cases for your request.\n",
            "### 🔍 Issue Identified:",
            f"- {reason}\n",
            "### 📝 Recommended Actions:",
            "- Provide a detailed Business Requirements Document (BRD) with specific functional requirements",
            "- Include clear user stories with acceptance criteria",
            "- Specify expected system behaviors and validation rules",
            "- Add compliance requirements if applicable",
            "- Ensure all referenced features have detailed descriptions\n",
            "Once you provide the necessary details, I'll be able to generate comprehensive test cases for your project."
        ]
        return "\n".join(summary_lines)
    
    else:
        # Parse test case table
        test_count = current_testcases.count("\n|") - 2  # Subtract header and separator rows
        test_count = max(0, test_count)  # Ensure non-negative
        
        # Extract compliance rules if present
        compliance_ids = []
        if "Applied Compliance Rules" in current_testcases:
            # Extract compliance section
            try:
                compliance_section = current_testcases.split("Applied Compliance Rules")[1]
                # Simple extraction of items that look like compliance IDs
                import re
                compliance_ids = re.findall(r'[A-Z][A-Z0-9\-]+', compliance_section)
                compliance_ids = list(set(compliance_ids))[:10]  # Limit to 10 unique IDs
            except:
                pass
        
        summary_lines = [
            "## ✅ Test Cases Generated Successfully!\n",
            f"I've generated **{test_count} test cases** for your application.\n",
            "### 📋 Test Coverage:",
            "- Comprehensive test scenarios covering functional requirements",
            "- Positive, negative, and boundary test cases",
            "- Input validation and error handling scenarios"
        ]
        
        if compliance_ids:
            summary_lines.append(f"\n### 🔒 Compliance Rules Covered:")
            summary_lines.append(f"Your test cases include validation for **{len(compliance_ids)} compliance rules**:")
            for compliance_id in sorted(compliance_ids)[:5]:  # Show top 5
                summary_lines.append(f"- {compliance_id}")
        
        summary_lines.append("\n### 📥 Next Steps:")
        summary_lines.append("- Review the generated test cases")
        summary_lines.append("- Integrate with your test management system")
        summary_lines.append("- Execute test cases during your testing phase")
        summary_lines.append("\nAll test cases are ready for use! 🎉")
        
        return "\n".join(summary_lines)

@retry(
    wait=wait_random_exponential(multiplier=2, max=60),  # Wait 2s, 4s, 8s... up to 60s
    stop=stop_after_attempt(10),  # Retry 10 times before failing
    retry=retry_if_exception_type(ResourceExhausted)  # Only retry on 429 errors
)
async def parse_testcases_to_json(current_testcases: str, model_name: str = "gemini-2.0-flash") -> dict:
    """
    Parses markdown table test cases into structured JSON format using Vertex AI.
    
    Args:
        current_testcases: Markdown table string containing test cases OR error message
        model_name: Model identifier (default: gemini-2.0-flash)
        
    Returns:
        Dictionary with parsed test cases and compliance information.
        Returns empty lists if current_testcases indicates generation failure.
    """
    
    parsing_prompt = f"""
You are a test case parser. Analyze the input and determine if it contains valid test cases or an error message.

**CRITICAL INSTRUCTION - Error Detection:**
Before attempting to parse, check if the input contains any of the following indicators:
- "cannot be generated"
- "unable to generate"
- "Test case enhancement cannot be generated"
- "insufficient information"
- "missing requirements"
- Any error or failure message explaining why test cases were not created

If ANY error indicator is present, return:
{{
  "testcase_id": "generate-random-uuid",
  "Testcase Title": "Test Cases Not Generated",
  "testcases": [],
  "compliance_ids": [],
  "error_message": "Extract the exact reason from the input"
}}

**ONLY IF the input contains a valid markdown table with test cases:**
Extract the following:

1. **Testcase Title**: Generate a concise title summarizing the test cases (max 10 words)
2. **Test Cases**: Extract each row into format [Sr.No, Test Description, Expected Result]
3. **Compliance Rules**: Extract all compliance rule IDs from the "Applied Compliance Rules" section

Input:
{current_testcases}

Return ONLY a valid JSON object in this exact format:
{{
  "testcase_id": "generate-random-uuid",
  "Testcase Title": "concise title here",
  "testcases": [
    ["1.", "Test description...", "Expected result..."],
    ["2.", "Another test...", "Expected result..."]
  ],
  "compliance_ids": ["HIPAA", "ISO 27001", "FDA", "ISO 9001"]
}}

Important:
- First check for error messages - if found, return empty lists immediately
- Extract Sr.No exactly as shown (including periods/dots)
- Keep test descriptions concise but complete
- Extract only the compliance rule names/IDs (e.g., "HIPAA", "SOC 2", "GDPR Article 32")
- Return ONLY the JSON, no explanations
"""
    
    try:
        # Pre-check: Detect error messages before calling LLM
        error_indicators = [
            "cannot be generated",
            "unable to generate",
            "test case enhancement cannot be generated",
            "insufficient information",
            "missing requirements",
            "insufficient requirements"
        ]
        
        contains_error = any(indicator in current_testcases.lower() for indicator in error_indicators)
        
        if contains_error:
            # Return immediately without parsing
            logger.warning("current_testcases contains error message, skipping parsing")
            return {
                "testcase_id": str(uuid.uuid4()),
                "Testcase Title": "Test Cases Not Generated",
                "testcases": [],
                "compliance_ids": [],
                "error_message": current_testcases.strip()
            }
        
        # Initialize Vertex AI Generative Model
        model = GenerativeModel(model_name)
        
        # Generate content
        response = model.generate_content(parsing_prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON response
        parsed_data = json.loads(response_text)
        
        # Generate UUID if not present or placeholder
        if "testcase_id" not in parsed_data or parsed_data["testcase_id"] == "generate-random-uuid":
            parsed_data["testcase_id"] = str(uuid.uuid4())
        
        # Log if test cases are empty (additional safety check)
        if "error_message" in parsed_data or len(parsed_data.get("testcases", [])) == 0:
            logger.warning(f"Test cases not generated: {parsed_data.get('error_message', 'Unknown reason')}")
            
        return parsed_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response text: {response_text}")
        raise
    except Exception as e:
        logger.error(f"Error parsing test cases: {e}")
        raise


# --- Configure logging to show output in the console ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestCaseProcessorAgent(BaseAgent):
    """
    An ADK agent that aggregates test cases. And handles logging and event authoring.
    """

    def __init__(self, name: str = "TestCaseProcessorAgent", **kwargs):
        """Initializes the agent."""
        super().__init__(name=name, **kwargs)
        # The self.logger attribute is no longer initialized here to prevent the error.

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the agent's logic with correct logging and event creation.
        """
        # --- Get the logger instance here ---
        logger = logging.getLogger(self.name)

        state = ctx.session.state
        state_delta: Dict[str, Any] = {}
        
        current_testcases = state.get("current_testcases")
        aggregated_testcases = state.get("aggregated_testcases", [])
        
        if aggregated_testcases is None:
            aggregated_testcases = []
        
        all_testcases_history = list(state.get("all_testcases_history", []))
        
        if current_testcases:
            # Parse the test cases using Vertex AI before appending
            try:
                parsed_json = await parse_testcases_to_json(
                    current_testcases, 
                    model_name="gemini-2.0-flash"  # or "gemini-2.5-pro" for better accuracy
                )
                logger.info(f"Successfully parsed test cases: {parsed_json['testcase_id']}")
                
                aggregated_testcases.append(parsed_json)
                all_testcases_history.append(parsed_json)
                
                summary_response = await summarize_testcases_from_markdown(current_testcases,)
                state_delta["final_summary"] = summary_response
                
            except Exception as e:
                logger.error(f"Failed to parse test cases: {e}")
                # Fallback: append error record
                aggregated_testcases.append({
                    "testcase_id": str(uuid.uuid4()),
                    "Testcase Title": "Parse Error",
                    "testcases": [],
                    "compliance_ids": [],
                    "raw_content": current_testcases,   
                    "error": str(e)
                })
        
        state_delta["aggregated_testcases"] = aggregated_testcases
        output_message = f"Aggregated {len(aggregated_testcases)} test case sets."    
        
            

        # If the loop is not finished, yield an event to update the state
        logger.info(output_message)
        yield Event(
            actions=EventActions(state_delta=state_delta),
            author=self.name
        )