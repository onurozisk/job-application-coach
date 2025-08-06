"""This code is part of a career advisor application that uses the IBM Watsonx AI service to provide
personalized career advice based on job descriptions and resumes.It utilizes
Gradio for the user interface and requires specific environment variables for API access.
"""
# 10/10 Pylint Static Code Analysis Score 
import os
from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

import gradio as gr


# Load environment variables from .env file
load_dotenv()

# Retrieve API key and project ID from environment variables
api_key = os.getenv("API_KEY")  # Ensure you have set this in your .env file
project_id = os.getenv("PROJECT_ID")  # Ensure you have set this in your .env file

# Model and project settings
MODEL_ID = (
    "meta-llama/llama-3-2-11b-vision-instruct"  # Directly specifying the LLAMA3 model
)

credentials = Credentials(url="https://eu-de.ml.cloud.ibm.com", api_key=api_key)

# Generation parameters
params = TextChatParameters(temperature=0.7, max_tokens=512)

model = ModelInference(
    model_id=MODEL_ID, credentials=credentials, project_id=project_id, params=params
)


def polish_resume(position_name, resume_content, polish_prompt=""):
    """Polish the resume content based on the position name and optional prompt.

    Args:
        position_name(string): The name of the position you are applying for.
        resume_content (string): The content of your resume.
        polish_prompt (string): Optional prompt to guide the polishing process.

    Returns:
        string: A polished version of the resume tailored to the position.
    """

    if polish_prompt and polish_prompt.strip():
        prompt_use = f"""Given the resume content: '{resume_content}',
        polish it based on the following instructions: {polish_prompt} 
        for the {position_name} position."""
    else:
        prompt_use = f"""Suggest improvements for the following resume content:
        '{resume_content}' to better align with the requirements and expectations of a {position_name} position.
        Return the polished version, highlighting necessary adjustments for clarity, relevance, 
        and impact in relation to the targeted role."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_use},
            ],
        }
    ]
    generated_response = model.chat(messages=messages)
    # Extract and return the generated text
    generated_text = generated_response["choices"][0]["message"]["content"]

    return generated_text


polish_resume_interface = gr.Interface(
    fn=polish_resume,
    inputs=[
        gr.Textbox(
            label="Position Name",
            placeholder="Enter the position you are applying for...",
        ),
        gr.Textbox(
            label="Resume Content", placeholder="Paste your resume content here..."
        ),
        gr.Textbox(
            label="Polish Prompt",
            placeholder="Optional: Provide specific instructions for polishing",
        ),
    ],
    outputs=gr.Textbox(
        label="Polished Resume", placeholder="Your polished resume will appear here..."
    ),
    title="Resume Polisher",
    description="""Upload your resume and specify the position you are applying for.
    The model will polish your resume to better fit the job description.""",
)

polish_resume_interface.launch()
