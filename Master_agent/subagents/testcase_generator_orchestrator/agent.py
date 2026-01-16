
from google.adk.agents import LoopAgent, SequentialAgent

from .subagents.testcase_generator_agent import testcase_generator_agent
from .subagents.requirement_analyst import testcase_requirements_generator
from .subagents.generated_testcase_collector import testcase_collector
from .subagents.feature_manager import feature_manager
from .subagents.feature_manager.TestCaseProcessorAgent import TestCaseProcessorAgent


# Create the Testcase Generator Loop Agent
testcase_generator_loop = LoopAgent(
    name="TestcaseGeneratorLoop",
    max_iterations=5,  
    sub_agents=[    
        testcase_generator_agent,
        TestCaseProcessorAgent(),
    ],
    description="Iteratively generates Testcase until all features have been processed",
)

new_testcase_generator = SequentialAgent(
    name="TestcaseGenerationPipeline",
    sub_agents=[
        testcase_requirements_generator,  # Step 1: Generate Testcase requirements
        testcase_generator_loop,  # Step 2: Generate Testcase in a loop
    ],
    description="Generates and refines a Testcase through an iterative review process",
)