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

# Set credentials to use the model
credentials = Credentials(url="https://eu-de.ml.cloud.ibm.com", api_key=api_key)

# Generation parameters
params = TextChatParameters(temperature=0.7, max_tokens=1024)


# Initialize the model
model = ModelInference(
    model_id=MODEL_ID, credentials=credentials, project_id=project_id, params=params
)


# Function to generate career advice
def generate_career_advice(position_applied, job_description, resume_content):
    """
    generate_career_advice function provides personalized advice on how to improve a resume
    based on the job description and the position applied for.

    Args:
        postition_applied(string): Name of the position you are applying for.
        job_description (string): The job description for the position.
        resume_content (string): The content of your resume.

    Returns:
        string: Advice on how to improve the resume for the applied position.
    """
    # The prompt for the model
    prompt = f"""Considering the job description: {job_description},
    and the resume provided: {resume_content},
    identify areas for enhancement in the resume. 
    Offer specific suggestions on how to improve these aspects to better match
    the job requirements and increase the likelihood of being selected for the
    position of {position_applied}."""

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

    # Extract and format the generated text
    advice = generated_response["choices"][0]["message"]["content"]
    return advice


# Create Gradio interface for the career advice application
career_advice_app = gr.Interface(
    fn=generate_career_advice,
    flagging_mode="never",  # Deactivate the flag function in gradio as it is not needed.
    inputs=[
        gr.Textbox(
            label="Position Applied For",
            placeholder="Enter the position you are applying for...",
        ),
        gr.Textbox(
            label="Job Description Information",
            placeholder="Paste the job description here...",
            lines=10,
        ),
        gr.Textbox(
            label="Your Resume Content",
            placeholder="Paste your resume content here...",
            lines=10,
        ),
    ],
    outputs=gr.Textbox(label="Advice"),
    title="Career Advisor",
    description="""Enter the position you're applying for, paste the job description,
    and your resume content to get advice on what to improve for getting this job.""",
)

# Launch the application
career_advice_app.launch()
