from langchain_postgres import PGVector

connection = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"  
collection_name = "scholar2"

from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

from enum import Enum
from typing import Optional, Annotated
from langchain.tools import tool

class JobType(str, Enum):
    parttime = "parttime"
    fulltime = "fulltime"
    negotiation = "negotiation"

class Position(str, Enum):
    Management = "Management/Leadership"
    Postdoc = "Postdoc Position"
    Teaching = "Teaching/Lecturer Position"
    Research = "Research Position"
    PhD = "PhD Scholarship"
    ProfessorTrack = "Assistant/Associate/Full Professor"
    Staff = "Staff/Technician/Engineer Position"
    Undergrad = "Undergraduate Scholarship"
    Other = "Other"
    Master = "Master Scholarship"
    Admin = "Administration/Managerment"
    Lecturer = "Lecturer Position"
    Professor = "Professor Position"
    Faculty = "Faculty Position"


@tool
def job_search_by_query(job: str, k: int = 3,
               job_type: Optional[JobType] = None,
               position: Optional[Position] = None) -> str:
    """
    Search for jobs based on a query with optional filters.

    Parameters:
    - job (str): The job title or description to search for.
    - k (int): The number of top similar jobs to retrieve.
    - job_type (Optional[JobType]): 'parttime' or 'fulltime' or 'negotiation'.
    - position (Optional[Position]): The job category to filter.

    Returns:
    - list: The job list after the search is invoked.
    """
    print("--tool 3: query pg--")
    print(job, k, job_type, position)

    filter = {}
    if job_type:
        filter["workingtime"] = job_type.value
    if position:
        filter["position"] = position.value

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            'score_threshold': 0.5,
            "k": k,
            'filter': filter
        }
    )
    
    return retriever.invoke(job)


from langgraph.prebuilt import InjectedState

@tool
def job_search_by_cv(cv: Annotated[str, InjectedState("cv")], 
                     k: int = 3,
               job_type: Optional[JobType] = None,
               position: Optional[Position] = None) -> str:
    """
    Search for jobs based on a provided curriculum vitae with optional filters.

    Parameters:
    - k (int): The number of top similar jobs to retrieve.
    - job_type (Optional[JobType]): 'parttime' or 'fulltime' or 'negotiation'.
    - position (Optional[Position]): The job category to filter.

    Returns:
    - list: The job list after the search is invoked.
    """
    print("--tool 2: cv pg--")
    print(k, job_type, position)
    filter = {}
    if job_type:
        filter["workingtime"] = job_type.value
    if position:
        filter["position"] = position.value

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            'score_threshold': 0.5,
            "k": k,
            'filter': filter
        }
    )
    
    return retriever.invoke(cv)