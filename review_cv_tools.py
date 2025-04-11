from langchain_together import ChatTogether
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
import dotenv
dotenv.load_dotenv()
import os

class Feedback(BaseModel):
    issue: str = Field(description="The issue identified by the human reviewer")
    solution: str = Field(description="The solution to the issue identified by the human reviewer")
    criteria: str = Field(description="The criteria used to evaluate the CV")

class Feedbacks(BaseModel):
    feedbacks: list[Feedback] = Field(description="The feedbacks from the human reviewers")

class SuggestChangeState(TypedDict):
    candidate_cv: str
    job_description: str
    review: list[Feedback]
    new_cv: str
    human_feedback: str
    

review_instruction = """
You are an experienced CV Reviewer & Recruitment Specialist responsible for evaluating a candidate’s CV 
based on a Job Description (JD) and 10 key hiring criteria.

For you, every CV has areas that need improvement—your job is to identify these weaknesses and 
provide actionable solutions to enhance the candidate’s profile. 
Your review should be both constructive and insightful, helping recruiters and candidates understand key strengths, 
weaknesses, and areas for development.

Your review must include at least 5 pieces of feedback across all criteria, ensuring a thorough evaluation of the candidate’s profile.
⸻

Context:

Job Description (JD):

{job_description}

Candidate’s CV:

{candidate_cv}

⸻

Carefully review the CV against the following 10 key criteria:

1, Job Fit
	•	Does the candidate’s experience align with the job requirements?
	•	Are there any major gaps between the JD and their CV?

2, Work Experience
	•	How relevant and extensive is their work experience?
	•	Do their previous roles demonstrate progression and growth?

3, Technical & Soft Skills
	•	Do they have the core skills required for the role?
	•	Are there any missing or underdeveloped skills?

4, Achievements & Impact
	•	Has the candidate demonstrated measurable success in past roles?
	•	Are there any specific metrics, outcomes, or contributions mentioned?

5, Education & Certifications
	•	Do they have the necessary academic background?
	•	Are there any relevant certifications that add value to their profile?

6, Consistency & Accuracy
	•	Does their experience and timeline make sense?
	•	Are there any inconsistencies or potential exaggerations?

7, CV Formatting & Readability
	•	Is the CV well-structured, clear, and professional?
	•	Are there any grammar, spelling, or formatting issues?

8, Projects & Contributions
	•	Has the candidate worked on any significant personal or professional projects?
	•	Do they showcase innovation, problem-solving, or leadership?

9, Cultural Fit
	•	Does their profile suggest they would integrate well into the company culture?
	•	Have they worked in similar environments before?

10, Growth Potential & Initiative
	•	Does the CV reflect a commitment to continuous learning and career progression?
	•	Has the candidate taken proactive steps to improve their skills?


Follow these instructions carefully:
	•	Be objective, structured, and insightful in your analysis.
	•	Every CV has weaknesses—always point them out and suggest solutions.
	•	Generate at least 5 key feedback points covering different aspects of the CV.
	•	Use clear bullet points and avoid generic responses—always provide specific reasons for each strength/weakness.
	•	If the CV lacks crucial information, note it explicitly.
	•	Base all assessments on the 10 key criteria provided.
"""

def suggest_cv(state: SuggestChangeState):
    print('--suggest--')
    candidate_cv = state["candidate_cv"]
    job_description = state["job_description"]
    # human_feedback = state.get("human_feedback","")

    system_message = review_instruction.format(
        candidate_cv=candidate_cv,
        job_description=job_description,
        # human_feedback=human_feedback
    )
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0,
        api_key=os.environ["TOGETHER_API_KEY"]

    )
    structured_llm = llm.with_structured_output(Feedbacks)

    feedbacks = structured_llm.invoke(
        [SystemMessage(system_message)] + [HumanMessage('Let start the review process')]
    )
	
    return {'review': feedbacks.feedbacks}

builder_1 = StateGraph(SuggestChangeState)
builder_1.add_node('suggest_cv', suggest_cv)
builder_1.set_entry_point('suggest_cv')
builder_1.set_finish_point('suggest_cv')
suggest_agent = builder_1.compile()



adjust_instruction = """

You are an AI-powered CV Editing Specialist. Your task is to improve and refine a candidate’s CV based on feedback provided by a CV Reviewer. Your goal is to enhance clarity, structure, and relevance while ensuring alignment with the job description (JD) and industry best practices.

You must fix weaknesses, integrate suggested improvements, and optimize the CV for better readability, impact, and job relevance while maintaining a professional tone.

⸻

Context:

Job Description (JD):

{job_description}

Original Candidate’s CV:

{candidate_cv}

⸻
Carefully adjust the CV following {n_keys} key criteria:

{criteria}

⸻

Follow these instructions carefully:
	•	Apply all feedback systematically and ensure significant improvements.
	•	Keep the CV concise, impactful, and optimized for recruiters and ATS systems.
	•	Ensure at least 5 major improvements based on the review.
	•	Use a professional and confident tone.
	•	Never fabricate information—only rephrase and enhance existing content.

"""
	# •	Ensure all the newly adjusted detail are in markdown formating green color. For instance, if TensorFlow was added to 'Frameworks: Pytorch' to become 'Frameworks: Pytorch, TensorFlow', then Tensorflow must be green highlight.

class ReviewedCV(BaseModel):
    new_cv: str = Field(description="The new suitalbe CV after reviewing the candidate's current CV")


def adjust_cv(state) -> ReviewedCV:
    print('--adjust--')

    candidate_cv = state["candidate_cv"]
    job_description = state["job_description"]
    feedbacks = state['review']
    criteria = "\n".join([f'{i+1}, {feedback.criteria}: {feedback.issue}\n\tSolution: {feedback.solution}' for i, feedback in enumerate(feedbacks)])

    system_message = adjust_instruction.format(
        job_description = job_description,
        candidate_cv = candidate_cv,
        criteria = criteria,
        n_keys = len(feedbacks)
    )
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0,
        api_key=os.environ["TOGETHER_API_KEY"]

    )
    structured_llm = llm.with_structured_output(ReviewedCV)

    new_cv = structured_llm.invoke(
        [SystemMessage(system_message)] + [HumanMessage('Let start the adjust process')]
    )
    return {'new_cv': new_cv.new_cv}

builder_2 = StateGraph(SuggestChangeState)
builder_2.add_node('adjust_cv', adjust_cv)
builder_2.add_edge(START, 'adjust_cv')
builder_2.add_edge('adjust_cv', END)

adjust_agent = builder_2.compile()



# combine 2 agent into 1

workflow = StateGraph(SuggestChangeState)
workflow.add_node('suggest', suggest_agent)
workflow.add_node('change', adjust_agent)
workflow.add_edge("suggest", "change")
workflow.set_entry_point("suggest")
workflow.set_finish_point('change')

review_agent = workflow.compile()



from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from langgraph.types import Command

@tool
def review_cv(job_description: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Review curriculum vitae by comparing itself with job description
    Args:
        job_description (str): String representing the job description.
    """

    print("--tool 3: review--")

    candidate_cv = state.get("cv", "")

    if not job_description:
        job_description = state.get('jd', "")
    
    if not candidate_cv:
        raise FileExistsError('CV is not uploaded yet.')
    if not job_description:
        raise FileExistsError('JD is not uploaded yet.')

    result = review_agent.invoke(
        {
            "job_description": job_description,
            "candidate_cv": candidate_cv,
        }
    )
    # return result
    # tool_call_id = state['messages'][-1].tool_calls[0]['id']
    return Command(
        update = {"messages": [ToolMessage(result, tool_call_id=tool_call_id)], "new_cv": result.get("new_cv", "")})
