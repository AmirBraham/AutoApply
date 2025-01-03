import json
import re
from typing import Annotated, TypedDict, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_mistralai import ChatMistralAI
import asyncio
import os
import base64
import nest_asyncio
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from enum import Enum


from utils import analyze_image_with_pixtral, clean_html, extract_elements_with_xpaths

# Enable nested asyncio for Jupyter compatibility
nest_asyncio.apply()


class JobApplicationState(TypedDict):
    """State for the job application workflow"""
    url: str
    html_page_content: str
    xpath_page_content: str
    screenshot_base64: str
    success: bool
    next_action: str
    is_complete: bool


class Actions(Enum):
    button_click = "button_click"
    close_page = "close_page"
    other = "other"


class PageAction(TypedDict):
    """Page action for the job application workflow"""
    action: Actions = Field(description="The action to perform")
    target: str = Field(description="The target element to interact with (ID or classname of the element)")


class AutoJobApplicant:
    def __init__(self, model_name: str = "mistral-small-latest"):
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
        self.graph.add_node("perform_action", self.perform_action)
        # Define the workflow
        
        self.graph.set_entry_point("navigate")
        self.graph.add_edge('navigate', 'analyze_page')
        
        self.graph.add_conditional_edges (
            'analyze_page', lambda state: "perform_action" if not state["is_complete"] else None)
        self.graph.add_edge( "perform_action",'analyze_page')
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
            html_page_content = await self.page.content()
            # Clean html_page_content 
            #Remove any script tags
            html_page_content = clean_html(html_page_content)
            print(html_page_content)
            await self.page.set_viewport_size({"width": 1024, "height": 768})
            # Take screenshot and convert to base64
            screenshot_bytes = await self.page.screenshot(
                
            )
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode()

            return {
                "html_page_content": html_page_content,
                "screenshot_base64": screenshot_base64
            }
        except Exception as e:
            print(f"Error in navigate_to_job: {e}")
            raise

    async def analyze_job_page(self, state: JobApplicationState) -> JobApplicationState:
        """Analyze the job posting page"""
        try:
            analysis = await analyze_image_with_pixtral(state["screenshot_base64"], state["html_page_content"], os.getenv("MISTRAL_API_KEY"))
            analysis_prompt = f"""
            I am applying for a job and need assistance in filling out an application form. 
            Based on the provided screenshot and HTML content of the job posting page, identify the next best action to take, along with the corresponding XPath of the target element. 

            Instructions:
            1. If the current page appears to be the final submission page of the job application, return the following JSON:
            {{
                "action": "close_page",
                "label": None
            }}
            2. Otherwise, analyze the screenshot and the XPath representation of the page to determine the most logical action to proceed with.

            Inputs:
            - Screenshot of the page (current view):
            {analysis}
            - HTML representation of the page:
            {state['html_page_content']}

            Please provide your response in JSON format, indicating the action and corresponding XPath to follow.

            Answer:
            """

            structured_llm = self.llm.with_structured_output(schema=PageAction)

            response = structured_llm.invoke(analysis_prompt)
            action = response["action"]
            target = response["target"]
            if action == Actions.close_page:
                return {"is_complete": True}
            next_action = {
                "action": action,
                "target": target
            }
            return {"is_complete": False, "next_action": json.dumps(next_action) }
        except Exception as e:
            print(f"Error in analyze_job_page: {e}")
            raise

    async def perform_action(self, state: JobApplicationState) -> JobApplicationState:
        """Perform the specified action and check if application is complete"""
        try:
            next_action = json.loads(state["next_action"])
            print(next_action)
            action = next_action["action"]
            target = next_action["target"]

            
            locator = self.page.locator(f"xpath={target}")
            count = await locator.count()
            print(count)
            if count != 1:
                return state
            
            if action == Actions.button_click.value:
                print(f"Performing action: {action} on target: {target}")
                await self.page.locator(f"xpath={target}").click()


            # Wait for any navigation or loading
            await self.page.wait_for_load_state("networkidle")
            
            return state

        except Exception as e:
            print(f"Error in perform_action: {e}")
            raise


async def main():
    try:
        # Initialize the auto job applicant
        applicant = AutoJobApplicant()
        await applicant.initialize()  # Initialize async components

        # Initial state
        initial_state = {
            "messages": [],
            "url": "https://jobs.bendingspoons.com/positions/671201070d672a6fe45a0d7b?utm_source=website&utm_medium=hero_CTA&utm_content=careers_page&contractType=internship&student=true",
            "page_content": "",
            "form_data": {},
            "screenshot_base64": "",
            "success": False
        }

        # Run the workflow
        # Use ainvoke instead of invoke
        final_state = await applicant.workflow.ainvoke(initial_state)

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
