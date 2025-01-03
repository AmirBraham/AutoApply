import logging
import re
import sys
import base64
from typing import Dict
from mistralai import Mistral
from bs4 import BeautifulSoup, Tag
from vertexai.generative_models import SafetySetting, Part, Image


def encode_image_to_base64(image_path):
    """
    Encodes an image to a Base64 string.

    Args:
        image_path (str): The file path of the image to encode.

    Returns:
        str: The Base64-encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
            print(f"Image {image_path} successfully encoded to Base64.")
            return encoded_string
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while encoding the image: {e}")
        sys.exit(1)



async def analyze_image_with_gemini(gemini,encoded_image,html_content,previous_action):
    
    # Safety settings
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
    }
    attempt = 0
    max_retries = 5
    initial_delay = 1
    backoff_factor = 2
    delay = initial_delay
    # Define the detailed prompt
    prompt = f""" 
    You are a helpful assistant that helps job seekers to apply for jobs online.
    You are helping a user to apply for a job on a company's website.
    
    This is the previous action that the user took:
    {previous_action if previous_action else "No previous action"}
    
    
    Here's the HTML content of the webpage:
    {html_content}
    
    Analyze the webpage and extract the most logicial next action to take to proceed with the job application process.
    If you think you need to scroll down or interact with the page, please mention that in the response.
    Example of actions include:
    - Currently looking at a modal , dismiss it by clicking on the close button.
    - Fill out the application form starting with the Name field.
    - Upload resume in PDF format
    - Click on the "Submit" button to submit the application.
    
    
    PLEASE RETURN a BRIF description of the current state of the page and ONE ACTION ONLY.
    """
    
    image_bytes_1 = base64.b64decode(encoded_image)
    
    while attempt < max_retries:
        try:
            response = gemini.send_message(
                [ Part.from_image(image=Image.from_bytes(image_bytes_1)) ,Part.from_text(prompt)],
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            print("generated description successfully")
            return response.text

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise  # Re-raise other exceptions




async def analyze_image_with_pixtral(encoded_image, html_content,previous_action, mistral_api_key, image_format="jpeg"):
    """
    Sends the Base64-encoded image to Pixtral Vision for analysis.

    Args:
        encoded_image (str): The Base64-encoded string of the image.
        mistral_api_key (str): Your Mistral API key.
        image_format (str): The image format (e.g., "jpeg", "png").

    Returns:
        str: The analysis result from Pixtral Vision.
    """
    try:
        # Specify the model
        model = "pixtral-12b-2409"

        # Initialize the Mistral client
        client = Mistral(api_key=mistral_api_key)

    
        # Define the detailed prompt
        prompt = f""" 
        You are a helpful assistant that helps job seekers to apply for jobs online.
        You are helping a user to apply for a job on a company's website.
        
        This is the previous action that the user took:
        {previous_action if previous_action else "No previous action"}
        
        Analyze the webpage and extract the most logicial next action to take to proceed with the job application process.
        If you think you need to scroll down or interact with the page, please mention that in the response.
        Example of actions include:
        - Currently looking at a modal , dismiss it by clicking on the close button.
        - Fill out the application form starting with the Name field.
        - Upload resume in PDF format
        - Click on the "Submit" button to submit the application.
        
        
        PLEASE RETURN ONE ACTION ONLY.
        """
        
        print(f"Prompt: {prompt}")  


        # Define the messages for the chat
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/{image_format};base64,{encoded_image}"
                    }
                ]
            }
        ]

        # Get the chat response
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )

        # Extract and return the content of the response
        analysis = chat_response.choices[0].message.content
        return analysis

    except Exception as e:
        print(
            f"An error occurred while analyzing the image with Pixtral Vision: {e}")
        sys.exit(1)

def clean_html(text):
    soup = BeautifulSoup(text, 'html.parser')

    # Remove <script>, <style>, <meta>, <svg> tags
    for tag in soup(['script', 'style', 'meta', 'svg']):
        tag.decompose()

    # Remove DOCTYPE
    for item in soup.contents:
        if isinstance(item, type(soup.Doctype)):
            item.extract()

    return str(soup)

    return text
