# auto_apply_with_analysis_base64.py

import os
import sys
import argparse
import base64
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from mistralai import Mistral

def launch_chrome_and_capture(url, screenshot_path):
    """
    Launches Chrome, navigates to the URL, and takes a screenshot.

    Args:
        url (str): The URL to navigate to.
        screenshot_path (str): The file path to save the screenshot.
    """
    try:
        # Set up Chrome options
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--start-maximized")  # Start maximized
        chrome_options.add_argument("--disable-infobars")  # Disable infobars
        chrome_options.add_argument("--disable-extensions")  # Disable extensions
        # Uncomment the next line to run Chrome in headless mode
        # chrome_options.add_argument("--headless")

        # Initialize Chrome WebDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Navigate to the specified URL
        driver.get(url)
        print(f"Navigated to {url}")

        # Wait for the page to load completely
        driver.implicitly_wait(10)  # Waits up to 10 seconds for elements to load

        # Take a screenshot and save it
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
        
        # Retrieve the HTML content of the page
        html_content = driver.page_source
        print("HTML content retrieved.")
        return html_content

    except Exception as e:
        print(f"An error occurred while launching Chrome or taking a screenshot: {e}")
    finally:
        # Close the browser
        driver.quit()



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

def analyze_image_with_pixtral(encoded_image,html_content, mistral_api_key, image_format="jpeg"):
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

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments containing the URL.
    """
    parser = argparse.ArgumentParser(description="Automate job application with page analysis using Base64 image encoding.")
    parser.add_argument("url", type=str, nargs='?', help="The URL of the job application page.")

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    if args.url:
        url = args.url
    else:
        # If no URL is provided as an argument, prompt the user
        url = input("Enter the URL to navigate to: ").strip()
        if not url:
            print("No URL provided. Exiting.")
            sys.exit(1)

    # Define paths and API keys
    screenshot_path = "page_screenshot.png"
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    if not mistral_api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        sys.exit(1)

    # Step 1: Launch Chrome and capture screenshot
    html_content = launch_chrome_and_capture(url, screenshot_path)

    # Step 2: Encode screenshot to Base64
    base64_image = encode_image_to_base64(screenshot_path)

    # Step 3: Analyze image with Pixtral Vision
    analysis = analyze_image_with_pixtral(base64_image,html_content, mistral_api_key)

    # Step 4: Print the analysis result
    print("\n=== Pixtral Vision Analysis ===")
    print(analysis)

    # Optional: Remove the screenshot after analysis
    try:
        os.remove(screenshot_path)
        print(f"Deleted local screenshot: {screenshot_path}")
    except Exception as e:
        print(f"Could not delete screenshot: {e}")

if __name__ == "__main__":
        main()
