# Job Application Coach

This repository contains three simple AI demo applications, built as part of the **IBM AI Developer Professional Certificate** on Coursera. Each app implements one of the course tasks using Python and open-source ML libraries.

---

## Projects

### 1. `career_advisor`  
A simple career-advisor chatbot that recommends learning paths, job roles, and resources based on user input. Trained/fine-tuned on a small career dataset.

### 2. `customized_cover_letter`  
Generates tailored cover letters. You provide a job description and your profile; the app returns a polished, role-specific letter.

### 3. `resume_polisher`  
Takes a plain-text resume and applies NLP techniques to rephrase, reorganize, and highlight key skills and achievements.

---

## Getting Started

### Prerequisites

- **Python 3.11+**  
- **Docker** (optional, for containerized runs)  

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/onurozisk/job-application-coach.git
   cd job-application-coach
   ```
2. Creating Virtual Environment and Installing Requirements:
   Virtualenv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   Requirements:
   ```bash
   pip install --upgrade pip
   pip install -r requirement.txt
   ```
4. Running locally:
   # Career Advisor
   ```bash
   python career_advisor/src/main.py
   ```
   # Customized Cover Letter
   ```bash
   python customized_cover_letter/src/main.py
   ```
   # Resume Polisher
   ```bash
   python resume_polisher/src/main.py
   ```
