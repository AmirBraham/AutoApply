from typing import Annotated, TypedDict, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_mistralai import ChatMistralAI
from langchain.agents import AgentType, initialize_agent
import asyncio
import os
import base64
from PIL import Image
from io import BytesIO
import nest_asyncio
import requests
from enum import Enum
from playwright.async_api import async_playwright

from utils import analyze_image_with_pixtral

# Enable nested asyncio for Jupyter compatibility
nest_asyncio.apply()


class JobApplicationState(TypedDict):
    """State for the job application workflow"""
    messages: Annotated[List[Dict], add_messages]
    url: str
    page_content: str
    form_data: Dict[str, Any]
    screenshot_base64: str
    success: bool

class AutoJobApplicant:
    def __init__(self, model_name: str = "mistral-large-latest"):
        self.llm = ChatMistralAI(
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            streaming=False,
            temperature=0,
            model=model_name,
            verbose=True,
            cache=False,
            safe_mode=True,
        )
        self.async_browser = None
        self.toolkit = None
        self.tools = None
        self.agent = None
        self.graph = None
        self.workflow = None

    async def initialize(self):
        """Initialize all async components"""
        await self.setup_browser_tools()
        self.setup_graph()
        self.save_workflow_graph()

    async def setup_browser_tools(self):
        """Initialize Playwright browser and tools"""
        self.playwright = await async_playwright().start()
        self.async_browser = await self.playwright.chromium.launch(headless=False)
        self.page = await self.async_browser.new_page()
        

    def setup_graph(self):
        """Setup the LangGraph workflow"""
        self.graph = StateGraph(JobApplicationState)
        
        # Add nodes for each step of the process with async functions
        self.graph.add_node("navigate", self.navigate_to_job)
        self.graph.add_node("analyze_page", self.analyze_job_page)

        # Define the workflow
        self.graph.set_entry_point("navigate")
        self.graph.add_edge('navigate', 'analyze_page')
        
        self.workflow = self.graph.compile()

    def save_workflow_graph(self):
        """Save the workflow graph as a PNG file"""
        try:
            with open("workflow_graph.png", "wb") as f:
                f.write(self.workflow.get_graph().draw_mermaid_png())
            print("Workflow graph saved successfully")
        except Exception as e:
            print(f"Failed to save workflow graph: {e}")

    async def navigate_to_job(self, state: JobApplicationState) -> JobApplicationState:
        """Navigate to job posting and capture initial state"""
        try:
            print(f"Navigate to {state['url']}")
            await self.page.goto(state['url'])

            
            # Get current page content
            page_content = await self.page.content()
            
            # Take screenshot and convert to base64
            screenshot_bytes = await self.page.screenshot()
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode()

            return {
                "page_content": page_content,
                "screenshot_base64": screenshot_base64
            }
        except Exception as e:
            print(f"Error in navigate_to_job: {e}")
            raise

    async def analyze_job_page(self, state: JobApplicationState) -> JobApplicationState:
        """Analyze the job posting page"""
        try:
            analysis_prompt = f"""
            Analyze this job posting page and extract key information:
            1. Required fields for application
            2. Location of submit button
            3. Any specific requirements or qualifications
            
            Page content:
            {state['page_content']}
            """
            print("analyzing page")
            analysis = await analyze_image_with_pixtral(state["screenshot_base64"], state["page_content"], os.getenv("MISTRAL_API_KEY"))
            print(analysis)            
            return state
        except Exception as e:
            print(f"Error in analyze_job_page: {e}")
            raise

async def main():
    try:
        # Initialize the auto job applicant
        applicant = AutoJobApplicant()
        await applicant.initialize()  # Initialize async components
        
        # Initial state
        initial_state = {
            "messages": [],
            "url": "https://jobs.sanofi.com/en/job/-/-/2649/20331798528?source=LinkedIn",
            "page_content": "",
            "form_data": {},
            "screenshot_base64": "",
            "success": False
        }
        
        # Run the workflow
        final_state = await applicant.workflow.ainvoke(initial_state)  # Use ainvoke instead of invoke
        
        if final_state["success"]:
            print("Job application submitted successfully!")
        else:
            print("Job application process failed. Please check the logs.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        await applicant.async_browser.close()
        await applicant.playwright.stop()

if __name__ == "__main__":
    asyncio.run(main())