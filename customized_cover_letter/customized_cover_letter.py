"""Implementation of a customized cover letter generator using IBM Watsonx AI's Llama 3.2 model."""
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

# Set credentials to use the model
credentials = Credentials(url="https://eu-de.ml.cloud.ibm.com", api_key=api_key)

# Generation parameters
params = TextChatParameters(temperature=0.7, max_tokens=512)

# Initialize the model
model = ModelInference(
    model_id=MODEL_ID, credentials=credentials, project_id=project_id, params=params
)


def generate_cover_letter(company_name, position_name, job_description, resume_content):
    """ Generate a customized cover letter based on the provided company name, position name,
    job description, and resume content.

    Args:
        company_name(string): The name of the company you are applying to.
        position_name(string): The name of the position you are applying for.
        job_description (string): The job description for the position.
        resume_content (string): The content of your resume.

    Returns:
        string: A customized cover letter tailored to the job description and resume content.
    """


    prompt = f"""Generate a customized cover letter using the company name: {company_name},
    the position applied for: {position_name}, 
    and the job description: {job_description}.
    Ensure the cover letter highlights my qualifications and experience as detailed in the resume content: {resume_content}.
    Adapt the content carefully to avoid including experiences not present in my resume but mentioned in the job description. 
    The goal is to emphasize the alignment between my existing skills and the
    requirements of the role."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Generate a response using the model with parameters
    generated_response = model.chat(messages=messages)

    # Extract and return the generated text
    cover_letter = generated_response["choices"][0]["message"]["content"]

    return cover_letter


# Create Gradio interface for the cover letter generation application
cover_letter_app = gr.Interface(
    fn=generate_cover_letter,
    flagging_mode="never",  # Deactivate the flag function in gradio as it is not needed.
    inputs=[
        gr.Textbox(
            label="Company Name", placeholder="Enter the name of the company..."
        ),
        gr.Textbox(
            label="Position Name", placeholder="Enter the name of the position..."
        ),
        gr.Textbox(
            label="Job Description Information",
            placeholder="Paste the job description here...",
            lines=10,
        ),
        gr.Textbox(
            label="Resume Content",
            placeholder="Paste your resume content here...",
            lines=10,
        ),
    ],
    outputs=gr.Textbox(label="Customized Cover Letter"),
    title="Customized Cover Letter Generator",
    description="""Generate a customized cover letter by entering the company name, position name,
    job description and your resume.""",
)

# Launch the application
cover_letter_app.launch()
