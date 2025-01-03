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


from utils import analyze_image_with_pixtral, clean_html

# Enable nested asyncio for Jupyter compatibility
nest_asyncio.apply()


class JobApplicationState(TypedDict):
    """State for the job application workflow"""
    url: str
    html_page_content: str
    screenshot_base64: str
    success: bool
    next_action: str=None
    is_complete: bool


class Actions(Enum):
    button_click = "button_click"
    fill_field = "fill_field"
    close_page = "close_page"
    other = "other"

class PageAction(TypedDict):
    """Page action for the job application workflow"""
    action: Actions = Field(description="The action to perform")
    target: str = Field(description="The target element to interact with (ID or classname of the element)")
    backup_text: str = Field(description="The visible text content of the element as a fallback")



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
        self.graph.add_node("start_workflow", self.start_workflow)
        self.graph.add_node("navigate_to_page", self.navigate_to_page)
        
        self.graph.add_node("analyze_page", self.analyze_job_page)
        self.graph.add_node("perform_action", self.perform_action)
        # Define the workflow
        
        self.graph.set_entry_point("start_workflow")
        self.graph.add_edge('start_workflow','navigate_to_page')
        
        self.graph.add_edge('navigate_to_page', 'analyze_page')
        
        self.graph.add_conditional_edges (
            'analyze_page', lambda state: "perform_action" if not state["is_complete"] else None)
        self.graph.add_edge("perform_action",'navigate_to_page')
        self.workflow = self.graph.compile()
        
    def save_workflow_graph(self):
        """Save the workflow graph as a PNG file"""
        try:
            with open("workflow_graph.png", "wb") as f:
                f.write(self.workflow.get_graph().draw_mermaid_png())
            print("Workflow graph saved successfully")
        except Exception as e:
            print(f"Failed to save workflow graph: {e}")

    async def start_workflow(self, state: JobApplicationState) -> JobApplicationState:
        print(f"Navigate to {state['url']}")
        await self.page.set_viewport_size({"width": 1024, "height": 768})
        await self.page.goto(state['url'])
        try:
            await self.page.wait_for_load_state('networkidle',timeout=10000)
        except Exception as e:
            print(f"Error waiting for page load: {e}")
            pass    
        return state

    
    async def navigate_to_page(self, state: JobApplicationState) -> JobApplicationState:
        """Navigate to job posting and capture initial state"""
        try:
            try:
                await self.page.wait_for_load_state('networkidle',timeout=10000)
            except Exception as e:
                print(f"Error waiting for page load: {e}")
                pass   
            # Get current page content
            screenshot_bytes = await self.page.screenshot(full_page=True)
            print("took a screenshot")
            # Save screenshot to local
            with open("screenshot.png", "wb") as f:
                f.write(screenshot_bytes)
                
            html_page_content = await self.page.content()
            html_page_content = clean_html(html_page_content)
            # Take screenshot and convert to base64
            
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode()

            return {
                "html_page_content": html_page_content,
                "screenshot_base64": screenshot_base64,
            }
        except Exception as e:
            print(f"Error in navigate_to_page: {e}")
            raise

    async def analyze_job_page(self, state: JobApplicationState) -> JobApplicationState:
        """Analyze the job posting page"""
        try:
            
            # Pixtral analysis returns the most appropriate action to take
            analysis = await analyze_image_with_pixtral(state["screenshot_base64"], state["html_page_content"],state.get("next_action",None), os.getenv("MISTRAL_API_KEY"))
            print(analysis)
            
            analysis_prompt = f"""
            You are currently on a job application page.
            
            Based on the provided HTML content and the user instructions, identify the most relevant HTML element to interact with. 
            Return the best action to take, along with the corresponding selector that would uniquely identify the element.

            Instructions:
            {analysis}
            
            HTML content of the page:
            {state['html_page_content']}

            Return your response as a JSON object with these fields:
            - action: The action to take (e.g., "button_click", "close_page")
            - target: A CSS selector to find the element. Prefer IDs when available, then unique class names, then other attributes.
                    Make the selector as specific as needed to uniquely identify the element.
            - backup_text: The visible text content of the element (if any) as a fallback

            If this appears to be the final submission page, return:
            {{
                "action": "close_page",
                "target": null,
                "backup_text": null
            }}
            
            Answer:
            """

            structured_llm = self.llm.with_structured_output(schema=PageAction)

            response = structured_llm.invoke(analysis_prompt)
            print(response)
            action = response["action"]
            target = response["target"]
            if action == Actions.close_page:
                return {"is_complete": True}
            next_action = {
                "action": action,
                "target": target,
                "backup_text": response["backup_text"]

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
            backup_text = next_action.get("backup_text")
            
            print(f"Attempting to perform {action} using selector: {target}")
            
            # Try different strategies to find the element
            element = None
            
            # Strategy 1: Try CSS selector
            try:
                element = self.page.locator(target)
                count = await element.count()
                if count != 1:
                    element = None
                    print(f"CSS selector found {count} elements, trying alternative methods")
            except Exception as e:
                print(f"CSS selector failed: {e}")
                
            # Strategy 2: Try finding by text content if we have backup text
            if element is None and backup_text:
                try:
                    element = self.page.get_by_text(backup_text, exact=True)
                    count = await element.count()
                    if count != 1:
                        element = None
                        print(f"Text search found {count} elements")
                except Exception as e:
                    print(f"Text search failed: {e}")
            
            # Strategy 3: Try finding by role and text
            if element is None and backup_text:
                try:
                    element = self.page.get_by_role("button", name=backup_text)
                    count = await element.count()
                    if count != 1:
                        element = None
                        print(f"Role search found {count} elements")
                except Exception as e:
                    print(f"Role search failed: {e}")
            
            if element is None:
                print("Failed to find element using all available methods")
                return state
                
            if action == Actions.button_click.value:
                print(f"Clicking element")
                await element.click()
                try:
                    await self.page.wait_for_load_state('networkidle',timeout=5000)
                except Exception as e:
                    print(f"Error waiting for page load: {e}")
                    pass
                print("done clicking")
            elif action == Actions.fill_field.value:
                if "email" in target:
                    print(f"Filling email field")
                    await element.fill("amirbrahamm@gmail.com")
                else:
                    await element.fill("Amir Braham")

            # Wait for any navigation or loading
            
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
            "url": "https://jobs.ashbyhq.com/gorgias/b814b40f-ce91-466f-a75e-7130789c131d",
            "page_content": "",
            "form_data": {},
            "screenshot_base64": "",
            "success": False
        }

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
