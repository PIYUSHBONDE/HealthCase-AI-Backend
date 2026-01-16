import logging
import time
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
from typing import AsyncGenerator, Any, Dict, List
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

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
async def summarize_testcases_output(
    aggregated_testcases: List[Dict], 
    model_name: str = "gemini-2.0-flash"
) -> str:
    """
    Summarizes aggregated test cases into a user-friendly message.
    
    Args:
        aggregated_testcases: List of dictionaries containing parsed test cases with structure:
            [{
                "testcase_id": "uuid",
                "Testcase Title": "title",
                "testcases": [["1.", "description", "expected"]],
                "compliance_ids": ["COMP-ID1", "COMP-ID2"]
            }]
        model_name: Model identifier (default: gemini-2.0-flash)
        
    Returns:
        Formatted summary message for the user
    """
    
    # Calculate statistics
    total_testcase_sets = len(aggregated_testcases)
    total_individual_tests = sum(len(tc["testcases"]) for tc in aggregated_testcases)
    
    # Extract unique compliance IDs across all test cases
    all_compliance_ids = set()
    for tc in aggregated_testcases:
        all_compliance_ids.update(tc.get("compliance_ids", []))
    
    # Prepare structured data for the LLM
    testcase_summary = []
    for idx, tc in enumerate(aggregated_testcases, 1):
        testcase_summary.append({
            "set_number": idx,
            "title": tc["Testcase Title"],
            "test_count": len(tc["testcases"]),
            "compliance_rules": tc.get("compliance_ids", [])
        })
    
    summarization_prompt = f"""
You are a test case report generator. Create a clear, professional summary message for the user based on the generated test cases.


**Input Statistics:**
- Total Test Case Sets: {total_testcase_sets}
- Total Individual Test Cases: {total_individual_tests}
- Unique Compliance Rules Covered: {len(all_compliance_ids)}


**Test Case Sets Generated:**
{json.dumps(testcase_summary, indent=2)}


**All Compliance Rules:**
{json.dumps(list(all_compliance_ids), indent=2)}


**Instructions:**
Generate a concise, user-friendly summary message that:
1. Starts with a success confirmation
2. Mentions the total number of test cases and sets generated
3. Briefly lists each test case set with its title and count
4. Highlights compliance rules covered (if any)
5. Ends with a positive closing statement


**Formatting Requirements:**
- Use plain text only, no markdown formatting
- Do NOT use asterisks (*) or any special characters for formatting
- Do NOT use bold (**text**) or italic (*text*) markers
- Use simple line breaks and indentation for structure
- Keep the tone professional yet friendly
- Maximum length: 300 words
- Include emojis sparingly for visual appeal (optional)


Return ONLY the formatted summary message as plain text, no additional formatting symbols.

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
        return generate_fallback_summary(aggregated_testcases, total_individual_tests, all_compliance_ids)


def generate_fallback_summary(
    aggregated_testcases: List[Dict], 
    total_individual_tests: int, 
    all_compliance_ids: set
) -> str:
    """
    Generates a fallback summary message without using LLM.
    
    Args:
        aggregated_testcases: List of test case dictionaries
        total_individual_tests: Total count of individual test cases
        all_compliance_ids: Set of unique compliance IDs
        
    Returns:
        Formatted summary string
    """
    summary_lines = [
        "## ✅ Test Cases Generated Successfully!\n",
        f"I've generated **{total_individual_tests} test cases** across **{len(aggregated_testcases)} test case set(s)**.\n",
        "### 📋 Generated Test Case Sets:\n"
    ]
    
    for idx, tc in enumerate(aggregated_testcases, 1):
        test_count = len(tc["testcases"])
        title = tc["Testcase Title"]
        summary_lines.append(f"{idx}. **{title}** - {test_count} test cases")
    
    if all_compliance_ids:
        summary_lines.append(f"\n### 🔒 Compliance Rules Covered:")
        summary_lines.append(f"Your test cases include validation for **{len(all_compliance_ids)} compliance rules**:")
        for compliance_id in sorted(all_compliance_ids):
            summary_lines.append(f"- {compliance_id}")
    
    summary_lines.append("\n### 📥 Next Steps:")
    summary_lines.append("- Review the generated test cases")
    summary_lines.append("- Export to your preferred format")
    summary_lines.append("- Integrate with your test management system")
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
        current_testcases: Markdown table string containing test cases
        model_name: Model identifier (default: gemini-2.0-flash)
        
    Returns:
        Dictionary with parsed test cases and compliance information
    """
    
    parsing_prompt = f"""
You are a test case parser. Extract the following from the markdown table:


1. **Testcase Title**: Generate a concise title summarizing the test cases (max 10 words)
2. **Test Cases**: Extract each row into format [Sr.No, Test Description, Expected Result]
3. **Compliance Rules**: Extract all compliance rule IDs from the "Applied Compliance Rules" section
4. **Source Document**: Extract the source document name from the "Source Requirement Document" section


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
  "compliance_ids": ["HIPAA", "ISO 27001", "FDA", "ISO 9001", "BRD_Healthcare_App_v1.0.pdf"]
}}


Important:
- Extract Sr.No exactly as shown (including periods/dots)
- Keep test descriptions concise but complete
- Extract only the compliance rule names/IDs (e.g., "HIPAA", "SOC 2", "GDPR Article 32")
- Include the source requirement document name from the "Source Requirement Document" section in the compliance_ids list
- Return ONLY the JSON, no explanations
Ensure the JSON is well-formed and can be parsed without errors.
"""
    
    try:
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


def get_feature_list(state):
    requirements = state.get("requirements", {"features_to_process": [] })
    features = requirements.get("features_to_process", [])
    # Handle if accidentally stored as JSON string
    if isinstance(features, str):
        try:
            features = json.loads(features)
        except Exception:
            # If it's not valid JSON, fallback to treating as plain string (unlikely, log or raise)
            features = []
    # At this point, features is always a list
    return features


class TestCaseProcessorAgent(BaseAgent):
    """
    An ADK agent that processes features, aggregates test cases, and terminates
    a loop. This version correctly handles logging and event authoring.
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
        output_message = "Processed a feature and updated test cases."

        requirements = state.get("requirements", {"features_to_process": [] })
        features_to_process = requirements.get("features_to_process", [])
        try:
            features_to_process = json.loads(features_to_process)
            logger.info(f"Current session state on invocation:\n{features_to_process}")
        except Exception:
            logger.error(f"Failed to parse features_to_process string as JSON list{features_to_process}")


        # If the processing list is empty, terminate the loop.
        if not features_to_process:
            logger.info(f"Current session state on invocation:\n{state}")
            logger.info("No features left to process. Terminating loop.")
            yield Event(actions=EventActions(escalate=True), author=self.name)
            return

        # Process the first feature in the list
        features_to_process.pop(0)
        requirements["features_to_process"] = features_to_process
        state_delta["requirements"] = requirements

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
        state_delta["all_testcases_history"] = all_testcases_history
        
        # Clear the current_testcases variable for the next iteration
        state_delta["current_testcases"] = ""
        
        # Check for termination condition *after* processing
        if not features_to_process:
            output_message = "Processed the final feature. Terminating loop."
            logger.info(f"Current session state on invocation:\n{state}")
            logger.info(output_message)
            
            summarize_testcases = await summarize_testcases_output(aggregated_testcases)
            state_delta["final_summary"] = summarize_testcases
            
            yield Event(
                actions=EventActions(state_delta=state_delta, escalate=True),
                author=self.name
            )
            return

        # If the loop is not finished, yield an event to update the state
        logger.info(output_message)
        yield Event(
            actions=EventActions(state_delta=state_delta),
            author=self.name
        )