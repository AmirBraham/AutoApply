import sys
import base64
from mistralai import Mistral

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
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            print(f"Image {image_path} successfully encoded to Base64.")
            return encoded_string
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while encoding the image: {e}")
        sys.exit(1)
        
        
        

async def analyze_image_with_pixtral(encoded_image,html_content, mistral_api_key, image_format="jpeg"):
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
            "Analyze the following webpage. Provide a detailed description of its layout, "
            "including all UI elements such as buttons, input fields, scroll views, images, "
            "and their exact positions or coordinates on the page. For each element, specify its type, "
            "label (if any), and coordinates (e.g., top-left and bottom-right corners). "
            "Additionally, consider the HTML structure provided to enhance the accuracy of the analysis. "
            "Present the information in a structured JSON format."
            "At the end of your analysis , provide an action plan which button to click to continue to the job application on how you would apply for the job."
        )

        # Combine prompt, HTML, and image data
        content_str = (
            f"{prompt}\n\n"
            f"HTML Content:\n{html_content}\n\n"
        )
        
        
        # Define the messages for the chat
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content_str
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
        print(f"An error occurred while analyzing the image with Pixtral Vision: {e}")
        sys.exit(1)
        
        