from typing import List, Annotated
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
import operator
from langgraph.constants import Send
from langchain_together import ChatTogether
from langchain_core.tools.base import InjectedToolCallId
from pydantic import BaseModel, Field
from typing import List, Optional
import dotenv
dotenv.load_dotenv()
import os

class JobCriteriaComparison(BaseModel):
    job_responsibilities: str = Field(..., description="Key responsibilities listed in the job")
    technical_skills_tools: str = Field(..., description="Required technical skills or tools")
    years_of_experience: str = Field(..., description="Years of experience required")
    education_certifications: str = Field(..., description="Required education or certifications")
    soft_skills: str = Field(..., description="Required soft skills or personality traits")
    industry_sector: str = Field(..., description="Industry or sector the job belongs to")
    location_mode: str = Field(..., description="Location and work mode: Remote / Hybrid / On-site")
    salary_range: Optional[str] = Field(None, description="Salary range if mentioned")
    career_growth: Optional[str] = Field(None, description="Mention of career growth or advancement opportunities")
    unique_aspects: Optional[str] = Field(None, description="Any unique benefits or characteristics of the job")


class AnalyzeState(MessagesState):
    jd: str
    jds: List[str]
    jd_analysis: Annotated[list, operator.add] # Send() API key

def do_nothing(state):
    pass

def router(state):
    print("--router--")
    return [
        Send("extract", {"jd": jd}) for jd in state.get("jds", [])
    ]


###################
analyze_instruction = """Analyze the following job description:
{jd}"""
def extract_agent(state):
    print("--sparse--")

    jd = state.get("jd", "")
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0,
        api_key=os.environ["TOGETHER_API_KEY"]
    )
    llm = llm.with_structured_output(JobCriteriaComparison)
    response = llm.invoke([SystemMessage(analyze_instruction.format(jd = jd)), HumanMessage("Conduct scoring")])
    return {"jd_analysis": [response]}


######################
summarize_instruction = """You are a hiring analyst AI assistant. Your task is to summarize and synthesize multiple job descriptions (JDs) that share the same job title.

Each JD has been analyzed based on a set of common criteria such as responsibilities, required skills, experience, education, and soft skills.

Your goal is to:

1. Identify and list **the most common patterns** shared across the JDs (e.g., most frequently listed responsibilities, common tools/technologies, typical years of experience required).
2. Point out **notable differences or variations** across the JDs (e.g., range of experience, differing skill requirements, industry-specific needs).
3. Highlight any **unique or standout aspects** in any of the JDs (e.g., special benefits, unusual requirements).
4. Optionally, categorize the JDs into **clusters or types** based on similarities (e.g., junior vs senior, industry type, remote vs on-site).
5. Conclude with a short **summary insight**: what does this tell us about the job market for this role?

Use markdown formatting. If appropriate, present findings in bullet points or tables for clarity."""

def summarize_agent(state):
    print("--summa--")

    jd_analysis = state.get("jd_analysis", "")
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0,
        api_key=os.environ["TOGETHER_API_KEY"]

    )
    response = llm.invoke([SystemMessage(f"Summarize the evaluations and provide final feedback"), HumanMessage(f"Here the analysis of jobs: {jd_analysis}")])
    return {"messages": [response]}



analyze_graph = StateGraph(AnalyzeState)
analyze_graph.add_node("_init", do_nothing)
analyze_graph.add_node("extract", extract_agent)
analyze_graph.add_node("summarize", summarize_agent)

analyze_graph.set_entry_point("_init")
analyze_graph.add_conditional_edges("_init", router, ["extract"])

analyze_graph.add_edge("extract", "summarize")
analyze_graph.set_finish_point("summarize")

analyze_agent = analyze_graph.compile()

# from IPython.display import Image, display
# try:
#     display(Image(analyze_agent.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     pass



from langchain_core.tools import tool

@tool
def compare_jobs_tool(jds: list[str], tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Analyze and summarize a list of job descriptions (JDs) sharing the same job title.

    This tool performs the following:
    - Evaluates each job description across multiple criteria (responsibilities, required skills, experience, education, soft skills).
    - Identifies common patterns shared across the JDs.
    - Highlights notable differences or unique features among them.
    - Optionally clusters JDs into categories (e.g., senior vs junior roles, industry types).
    - Produces a high-level insight into the job market demand and expectations for the role.

    Args:
        jds (list[str]): A list of raw job descriptions, all referring to the same job title.

    Returns:
        A structured summary including synthesized insights, markdown tables or bullet lists
        capturing trends, variations, and recommendations based on the input JDs.
    """
    print("--tool7: jobs analysis--")
    response = analyze_agent.invoke({"jds": jds})
    return response
    # return Command(update ={
    #     "messages": [ToolMessage(response, tool_call_id)]
    # })
