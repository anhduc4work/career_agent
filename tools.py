from retrieve_tools import search_by_cv, search_by_query
from review_cv_tools import review_cv
from parser_tools import update_cv_from_chat, update_jd_from_chat
from score_jd_tools import score_jobs
from analyze_market_tools import compare_jobs_tool

tools = [search_by_query, search_by_cv, review_cv, update_cv_from_chat, update_jd_from_chat, score_jobs, compare_jobs_tool]