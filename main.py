import json
import re
from typing import  List, TypedDict
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
import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel

from utils import analyze_image_with_gemini, analyze_image_with_pixtral, clean_html

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
    scroll_down = "scroll_down"
    fill_form = "fill_form"  # New action
    other = "other"

class PageAction(TypedDict):
    """Page action for the job application workflow"""
    action: Actions = Field(description="The action to perform")
    target: str = Field(description="The target element to interact with (ID or classname of the element)")
    backup_text: str = Field(description="The visible text content of the element as a fallback")



class FormField(BaseModel):
    field_id: str = Field(description="The ID or selector of the form field")
    field_type: str  = Field(description="The type of the form field")
    label: str = Field(description="The label or placeholder text of the form field")
    
    
class Form(BaseModel):
    """Represents a form with its fields"""
    form_id: str = Field(description="The ID or selector of the form element")
    fields: List[FormField] = Field(description="List of form fields")


class UserData(BaseModel):
    """User's personal information for job applications"""
    full_name: str = "Amir Braham"
    email: str = "amirbrahamm@gmail.com"
    phone: str = "+1234567890"
    linkedin: str = "linkedin.com/in/amirbraham"
    github: str = "github.com/amirbraham"
    portfolio: str = "amirbraham.com"
    education: str = "Bachelor's in Computer Science"
    years_experience: str = "5"
    current_company: str = "Tech Corp"
    current_role: str = "Software Engineer"


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
        self.gemini = None
        creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        secret = json.loads (creds_json)
        
        credentials = service_account.Credentials.from_service_account_info(secret)

        vertexai.init(project="gen-lang-client-0613559779",
              location="europe-west2",
              api_key=os.getenv("VERTEX_API_KEY"), credentials=credentials)
    async def initialize(self):
        """Initialize all async components"""
        self.gemini = GenerativeModel("gemini-1.5-pro-002").start_chat()
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
            screenshot_bytes = await self.page.screenshot(full_page=False)
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
            analysis = await analyze_image_with_gemini(self.gemini,state["screenshot_base64"], state["html_page_content"],state.get("next_action",None))
            print(analysis)
            
            analysis_prompt = f"""
            You are currently on a job application page.
            
            Based on the provided HTML content and the user instructions, identify if there's a form to fill or a specific element to interact with.
            If you see a form with multiple fields (like name, email, etc.), recommend the fill_form action.
            
            Instructions:
            {analysis}
            
            HTML content of the page:
            {state['html_page_content']}

            Return your response as a JSON object with these fields:
            - action: The action to take (e.g., "button_click", "fill_form", "close_page", "scroll_down")
            - target: For non-form actions, a CSS selector to find the element. For forms, use the form's container selector.
            - backup_text: The visible text content of the element (if any) as a fallback

            If this appears to be a form, prioritize returning:
            {{
                "action": "fill_form",
                "target": "form selector or container",
                "backup_text": "Form container text"
            }}
            
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
            if action == Actions.fill_form.value:
                # Analyze form fields within the target container
                form_container = self.page.locator(target)
                html_content = await form_container.inner_html()
                html_content = clean_html(html_content)
                response = await self.analyze_form_fields(html_content)
                form_id = response.form_id
                fields = response.fields
                # Fill each field in the form
                user_data = UserData()  # Your predefined user data
                for field in fields:
                    try:
                        element = self.page.locator(field.field_id)
                        value = await self.match_field_to_user_data(field, user_data)
                        
                        if value:
                            await element.fill(value)
                            print(f"Filled {field.label} with {value}")
                            await asyncio.sleep(0.5)  # Small delay between fields
                            
                    except Exception as e:
                        print(f"Error filling field {field.label}: {e}")
                        continue
            
                return state
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
            elif action == Actions.scroll_down.value:
                print(f"Scrolling down")
                await self.page.evaluate("window.scrollBy(0, 500)")
                

            # Wait for any navigation or loading
            
            return state

        except Exception as e:
            print(f"Error in perform_action: {e}")
            raise
    
        
    async def analyze_form_fields(self, html_content: str):
        """Analyze the HTML content to identify form fields"""
        
        analysis_prompt = f"""
        Analyze the following HTML content and identify all form fields.
        For each field, determine:
        1. The field ID or unique selector
        2. The field type (text, email, phone, etc.)
        3. The label or placeholder text
        4. Whether it's required
        
        HTML content:
        {html_content}
        
        Return the analysis as a JSON array of objects with these properties:
        - field_id: string (CSS selector to uniquely identify the field)
        - field_type: string
        - label: string
        - required: boolean
        """
        
        structured_llm = self.llm.with_structured_output(schema=Form)
        response = structured_llm.invoke(analysis_prompt)   
        print(response)
        return response


    async def match_field_to_user_data(self, field: FormField, user_data: UserData) -> str:
        """Match a form field to the appropriate user data"""
        
        matching_prompt = f"""
        Determine which user data field best matches this form field:
        
        Form field:
        - Label: {field.label}
        - Type: {field.field_type}
        
        Available user data fields:
        {[field for field in UserData.__annotations__]}
        
        Return just the name of the matching field, or "none" if no good match.
        """
        
        response = self.llm.invoke(matching_prompt)
        response = response.content
        field_name = response.strip().lower()
        
        if field_name in UserData.__annotations__:
            return getattr(user_data, field_name)
        return ""

    async def fill_form(self, state: JobApplicationState) -> JobApplicationState:
        """Fill out the job application form"""
        try:
            # Initialize user data
            user_data = UserData()
            
            # Analyze form fields
            form_fields = await self.analyze_form_fields(state["html_page_content"])
            
            # Fill each field
            for field in form_fields:
                try:
                    # Find the element
                    element = self.page.locator(field.field_id)
                    
                    # Get the appropriate user data
                    value = await self.match_field_to_user_data(field, user_data)
                    
                    if value:
                        await element.fill(value)
                        print(f"Filled {field.label} with {value}")
                        
                    # Wait briefly between fields to avoid triggering anti-bot measures
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error filling field {field.label}: {e}")
                    continue
            
            return state
            
        except Exception as e:
            print(f"Error in fill_form: {e}")
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
