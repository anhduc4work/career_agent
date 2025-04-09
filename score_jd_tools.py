from typing import List, Annotated
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import operator
from langchain_core.tools.base import InjectedToolCallId
from langgraph.constants import Send
from langchain_together import ChatTogether
from pydantic import BaseModel, Field, model_validator
from langgraph.types import Command

class CVJDMatchFeedback(BaseModel):
    job_title_relevance:     int = Field(..., ge=0, le=10, description="Score (0-10): How well does the candidate's experience align with the job title?")
    years_of_experience:     int = Field(..., ge=0, le=10, description="Score (0-10): Does the candidate have sufficient experience for the position?")
    required_skills_match:   int = Field(..., ge=0, le=10, description="Score (0-10): To what extent does the candidate possess the skills listed in the JD?")
    education_certification: int = Field(..., ge=0, le=10, description="Score (0-10): Does the candidate's academic background fit the job requirements?")
    project_work_history:    int = Field(..., ge=0, le=10, description="Score (0-10): Are the candidate’s past projects or roles relevant to this position?")
    softskills_language:     int = Field(..., ge=0, le=10, description="Score (0-10): Does the candidate show relevant communication, leadership, or other soft skills?")
    overall_comment: str = Field(..., description="One overall comment about the candidate’s fit for the job.")
    overall_fit_score:       float = Field(0, description="Average score (0-10) calculated from all score fields.")

    @model_validator(mode="after")
    def compute_overall_score(self):
        self.overall_fit_score = round((
            self.job_title_relevance +
            self.years_of_experience +
            self.required_skills_match +
            self.education_certification +
            self.project_work_history +
            self.softskills_language
        ) / 6, 2)
        return self
    
    

class ScoreState(MessagesState):
    cv: str
    jd: str
    jds: List[str]
    jd_analysis: Annotated[list, operator.add] # Send() API key

def router(state):
    print("--router--")
    print(type(state.get("jds", [])),state.get("jds", []))
    return [
        Send("score", {"jd": jd}) for jd in state.get("jds", [])
    ]

score_instruction = """Based on the following job description and curiculum vitae, evaluate the candidate's fit:
Job Description:
{jd}

CV:
{cv}"""
def score_agent(state):
    print("--score--")

    jd = state.get("jd", "")
    cv = state.get("cv", "")
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0,
    )
    llm = llm.with_structured_output(CVJDMatchFeedback)
    response = llm.invoke([SystemMessage(score_instruction.format(cv = cv, jd = jd)), HumanMessage("Conduct scoring")])
    return {"jd_analysis": [response]}


def summarize_score_agent(state):
    print("--summa--")

    jd_analysis = state.get("jd_analysis", "")
    print(jd_analysis)
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0,
    )
    response = llm.invoke([SystemMessage(f"Summarize the evaluations and provide final feedback"), HumanMessage(f"Here the analysis of jobs: {jd_analysis}")])
    return response

def do_nothing(state):
    pass


score_graph = StateGraph(ScoreState)
score_graph.add_node("init", do_nothing)
score_graph.add_node("score", score_agent)
score_graph.add_node("summarize", summarize_score_agent)

score_graph.set_entry_point("init")
score_graph.add_conditional_edges("init", router, ["score"])

score_graph.add_edge("score", "summarize")
score_graph.set_finish_point("summarize")

score_agent = score_graph.compile()

# from IPython.display import Image, display
# try:
#     display(Image(score_agent.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     pass

from langchain_core.tools import tool
@tool
def score_jobs(jds: list[str], cv: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Compare a list of job descriptions (JDs) against a single CV and evaluate the candidate's fit for each job.

    This tool scores the candidate's CV for each provided JD based on multiple criteria such as:
    - Job title relevance
    - Years of experience
    - Required skills match
    - Education and certifications
    - Project and work history relevance
    - Soft skills and language
    It also generates an overall comment for each match and returns a summarized evaluation.

    Args:
        jds (list[str]): A list of raw text strings representing the job descriptions.
        cv (str): A string containing the candidate's resume or CV text.

    Returns:
        A structured summary containing score evaluations and comments for each JD compared to the CV.
    """
    print("--tool6: score--")
    response = score_agent.invoke({"jds": jds, "cv": cv})
    return response