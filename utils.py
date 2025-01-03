import re
import sys
import base64
from typing import Dict
from mistralai import Mistral
from bs4 import BeautifulSoup, Tag


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


async def analyze_image_with_pixtral(encoded_image, html_content, mistral_api_key, image_format="jpeg"):
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
        prompt = (
            "Analyze the provided webpage's structure and layout. Describe in detail all UI elements, "
            "including buttons, input fields, scrollable areas, images, and other components. Specify their "
            "positions or coordinates on the page. If there are any modals, identify and describe them clearly. "
        )


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


def get_xpath(element: Tag) -> str:
    """
    Compute the XPath of a BeautifulSoup Tag.

    Args:
        element (Tag): A BeautifulSoup Tag object.

    Returns:
        str: The XPath string.
    """
    components = []
    child = element if element.name else element.parent
    for parent in child.parents:
        if parent.name == '[document]':
            break
        siblings = parent.find_all(child.name, recursive=False)
        if len(siblings) > 1:
            index = siblings.index(child) + 1
            components.append(f"{child.name}[{index}]")
        else:
            components.append(child.name)
        child = parent
    components.reverse()
    xpath = '/' + '/'.join(components)
    return xpath


def extract_elements_with_xpaths(html_content: str) -> Dict[str, str]:
    """
    Extract all HTML elements and their XPaths from the given HTML content.

    Args:
        html_content (str): The HTML content as a string.

    Returns:
        Dict[str, str]: A dictionary mapping element descriptions to their XPaths.
    """
    soup = BeautifulSoup(html_content, 'lxml')

    # Remove all <script> tags
    for script in soup(["script", "style"]):
        script.decompose()

    elements_with_xpaths = {}

    # Traverse all elements
    for element in soup.find_all(True):
        xpath = get_xpath(element)
        # Create a unique description for the element
        desc = f"<{element.name}>"
        if element.get('id'):
            desc += f"[@id='{element['id']}']"
        elif element.get('class'):
            classes = " ".join(element.get('class'))
            desc += f"[@class='{classes}']"
        else:
            # Optionally include text or other attributes
            text = element.get_text(strip=True)
            desc += f"[text()='{text[:30]}']"  # Truncate long texts
        elements_with_xpaths[desc] = xpath

    return elements_with_xpaths


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
