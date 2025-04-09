
from langchain_together import ChatTogether
from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from trustcall import create_extractor
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from langgraph.types import Command
from langchain_core.tools.base import InjectedToolCallId


@tool
def update_cv_from_chat(state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Update raw text Curriculum Vitae into state from messages"""
    print('--tool 4: sparse cv--')
    
    messages = state["messages"]
    if state.get("cv", ""):
        return "CV is already update"
    
    
    class CurriculumVitae(BaseModel):
        content: str = Field(description="full raw text of the CV")

    model = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0,
    )

    cv_extractor = create_extractor(
        model,
        tools=[CurriculumVitae],
        tool_choice="CurriculumVitae"
    )
    system_msg = "Extract the raw text content of Curriculum Vitae from the following conversation if it does exist"

    # Invoke the extractor
    result = cv_extractor.invoke({"messages": [SystemMessage(content=system_msg)]+messages})

    if len(result['responses'][0].content) < 100:
        return "User haven't upload it yet."

    return Command(
        update = {
            "messages": [ToolMessage(f"Successfully update CV from messages:\n{result['responses'][0].content}", tool_call_id=tool_call_id)],
            "cv" : result['responses'][0].content
        },
    )

@tool
def update_jd_from_chat(state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Update raw text desired job description into state if user show interest"""
    print('--tool 4: sparse jd--')

    messages = state["messages"]
    
    class JobDescription(BaseModel):
        content: str = Field(description="full raw text of the JD")
    model = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0,
    )
    jd_extractor = create_extractor(
        model,
        tools=[JobDescription],
        tool_choice="JobDescription"
    )

    system_msg = "Extract the raw text of JD that user interested in from the following conversation"

    # Invoke the extractor
    result = jd_extractor.invoke({"messages": [SystemMessage(content=system_msg)]+messages})
    return Command(
        update = {
            "messages": [ToolMessage(f"Successfully update JD from messages:\n{result['responses'][0].content}", tool_call_id=tool_call_id)],
            "jd" : result['responses'][0].content
        },
    )
