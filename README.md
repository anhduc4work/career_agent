
# Career Agent 🧑‍💼🤖

**Career Agent** is an AI-powered assistant designed to help job seekers find relevant job opportunities and improve their resumes. Leveraging advanced tools and models, it analyzes job descriptions, reviews resumes, and provides actionable insights to enhance your job search.

## 🚀 Features

- **Job Description Analysis**: Understand the key requirements and expectations of various job postings.
- **Resume Review**: Get feedback on your resume to better align with desired job roles.
- **Market Tools Analysis**: Evaluate and compare tools and technologies prevalent in the job market.
- **Interactive Interface**: Engage with the assistant through a user-friendly Gradio interface.

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/anhduc4work/career_agent.git
   cd career_agent
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💡 Usage

To start the application:

```bash
python app.py
```

This will launch the Gradio interface in your default web browser, allowing you to interact with the Career Agent.

## 🧰 Frameworks & Technologies Used

- **Python 3.10+**
- **LangGraph** – for building multi-agent workflows
- **Gradio** – for building the interactive UI
- **MongoDB** – used for storing checkpoints (via `langgraph.checkpoint`)
- **OpenAI API / Together API / Ollama** – for language/embedding models
- **Google Colab** – for testing

## 📁 Project Structure

- `app.py`: Main application file to launch the Gradio interface.
- `main_agent.py`: Core logic for the Career Agent's functionalities.
- `retrieve_tools.py`: Module to fetch and process job-related tools and technologies.
- `review_cv_tools.py`: Functions to analyze and provide feedback on resumes.
- `score_jd_tools.py`: Tools to evaluate and score job descriptions.
- `analyze_market_tools.py`: Analyze market trends and tools.
- `parser_tools.py`: Utility functions for parsing and processing data.
- `tools.py`: Additional helper functions.
- `TestColabVM.ipynb`: Jupyter notebook for testing in Google Colab.