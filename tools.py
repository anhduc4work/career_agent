# from retrieve_tools import search_by_cv, search_by_query
from review_cv_tools import review_cv
from parser_tools import update_cv_from_chat, update_jd_from_chat
from score_jd_tools import score_jobs
from analyze_market_tools import compare_jobs_tool
from recall_memory_tools import recall_memory
from retrieve_tool_pg import job_search_by_cv, job_search_by_query

tools = [
    job_search_by_cv, job_search_by_query,
    # search_by_query, search_by_cv, 
         recall_memory,
         review_cv, 
         update_cv_from_chat, update_jd_from_chat, 
         score_jobs, compare_jobs_tool]