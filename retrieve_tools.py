import dotenv
dotenv.load_dotenv()

# Init connection -> Database -> Collection
import pymongo
def get_mongo_client(mongo_uri):
    """Establish and validate connection to the MongoDB."""

    client = pymongo.MongoClient(
        mongo_uri
    )

    # Validate the connection
    ping_result = client.admin.command("ping")
    if ping_result.get("ok") == 1.0:
        # Connection successful
        print("Connection to MongoDB successful")
        return client
    print("Connection to MongoDB failed")
    return None


import os



# Embedding model
from langchain_ollama import OllamaEmbeddings



# Vector search
def vector_search(user_query, collection, limit):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    # print("base_url: ", base_url)
    embedding_model = OllamaEmbeddings(
        model = 'nomic-embed-text',
        base_url=base_url,
        # Maximum tokens for nomic-embed 8191-8192
    )
    # query_embedding = get_embedding(user_query)
    query_embedding = embedding_model.embed_query(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": limit,  # Return top k matches
        }
    }

    unset_stage = {
        "$unset": "embedding"  # Exclude the 'embedding' field from the results
    }

    project_stage = {
        "$project": {
            "_id": 0,  # Exclude the _id field
            "PositionList": 1,  # Include the title field
            "WorkingTime": 1,  # Include the title field
            "TextContent": 1,  # Include the genres field
            "score": {
                "$meta": "vectorSearchScore"  # Include the search score
            },
            "OriginalLink": 1,  # Include the plot field
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)



# Relevant tools 
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage


@tool
def search_by_query(query: str, limit: int = 5) -> str:
    """
    Searches for relevant open job positions by inputing job description.
    Should be used with short query.

    Args:
        query (str): String representing the job description to search for.
        limit (int): Number of results to return.
    """
    print("--tool 1: qr--")
    if not query:
        raise KeyError("query is not available")
    

    # MongoDB connection
    MONGO_URI = os.environ["MONGO_URI"]
    mongo_client = get_mongo_client(MONGO_URI)
    DB_NAME = "scholarship"
    JOB_COLLECTION_NAME = "job_description"
    db = mongo_client.get_database(DB_NAME)
    collection = db.get_collection(JOB_COLLECTION_NAME)
    # print(db, collection)

    get_knowledge = vector_search(query, collection, limit)
    search_results = []
    for result in get_knowledge:
        search_results.append(
            [
                result.get("score", "N/A"),
                result.get("PositionList", "N/A"),
                result.get("WorkingTime", "N/A"),
                result.get("TextContent", "N/A"),
                result.get("OriginalLink", "N/A"),

            ]
        )
    # close the MongoDB connection
    mongo_client.close()
    
    return search_results
   

@tool
def search_by_cv(state: Annotated[dict, InjectedState], limit: int = 5):
    """
    Searches for relevant open job positions by inputing job description.
    Should be used for query full CV text.
    Args:
        limit (int): Number of results to return.
    """
    print("--tool 2: cv--")
    query = state.get('cv', '')
    print("sender: ", state.get('sender', ''))

    if not query:
        raise FileExistsError('CV is not uploaded yet.')


    # MongoDB connection
    MONGO_URI = os.environ["MONGO_URI"]
    mongo_client = get_mongo_client(MONGO_URI)
    DB_NAME = "scholarship"
    JOB_COLLECTION_NAME = "job_description"
    db = mongo_client.get_database(DB_NAME)
    collection = db.get_collection(JOB_COLLECTION_NAME)

    get_knowledge = vector_search(query, collection, limit)
    search_results = []


    for result in get_knowledge:
        search_results.append(
            [
                result.get("score", "N/A"),
                result.get("TextContent", "N/A"),
            ]
        )

    # close the MongoDB connection
    mongo_client.close()

    return search_results
    