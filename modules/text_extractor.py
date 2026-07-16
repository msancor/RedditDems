from typing import List, Dict
from operator import add 
from pyspark import RDD
import pandas as pd
import regex as re
import tldextract
import hashlib
import joblib
import json

DATA_PATH = "data_shared/supplementary_data/"
REDDIT_DATA_PATH = "data_shared/reddit/"


def data_preprocessing(rdd:RDD, percentage=10) -> RDD:
    """
    This function preprocesses the data by performing the following steps:
    - Load the data from the Reddit submissions.
    - Filter the submissions with demographic scores available.
    - Filter the bots.
    - Filter the submissions with score greater than 1.
    - Count the number of submissions per author, subreddit and timestep.
    Args:
        rdd (RDD): The RDD containing the data.
        percentage (int): The percentage of users to sample.
    Returns:
        subreddit_processed (RDD): The RDD containing the preprocessed data.
    """
    AUTHOR_ID, SUBREDDIT_ID, SCORE_ID, TEXT_ID, URL_ID = 0, 1, 2, 3, 4
    subreddit_list = __get_subreddit_list()
    bots = __get_bot_list()
    
    subreddit_processed = (
        rdd
        .map(__resilient_json) #Load the JSON
        .map(lambda x: (x.get("author"), x.get("subreddit"), x.get("score"), x.get("selftext"), x.get("url"))) #Get the necessary fields
        .filter(lambda x: __hash_filter(x[AUTHOR_ID]) < percentage) #Sample the users
        .filter(lambda x: x[AUTHOR_ID] not in bots) #Filter the bots
        .filter(lambda x: x[SCORE_ID] > 1) #Filter the submissions with score greater than 1
        .filter(lambda x: x[SUBREDDIT_ID] in subreddit_list) #Filter the subreddits of interest
        )
    
    return subreddit_processed

def get_urls(processed_rdd:RDD) -> RDD:
    AUTHOR_ID, SUBREDDIT_ID, TEXT_ID, URL_ID, TIMESTAMP_ID, CATEGORY_ID = 0, 1, 2, 3, 4, 5
    prohibited_words = ["reddit", "redd.it", "imgur"]
    extracted_urls = (
        processed_rdd
        .flatMap(lambda x: [(x[AUTHOR_ID],x[SUBREDDIT_ID], word) for word in __extract_urls(x[TEXT_ID], x[URL_ID])])  # (author, subreddit, url)
        .filter(lambda x: all(word not in x[2] for word in prohibited_words)) #Filter the prohibited words
        .map(lambda x: (x[0], x[1], safe_urlparse(x[2]))) #Standardize the domains
        .filter(lambda x: x[2] is not None) #Filter the None values
        .map(lambda x: (x[0], x[1], "youtube.com" if x[2] == "youtu.be" else x[2]))
    )
    return extracted_urls

def count_urls(rdd:RDD, count_on="all") -> RDD:
    """
    This function counts the urls extracted in the text.
    Args:
        rdd (RDD): The RDD containing the data.
    Returns:
        word_count (RDD): The RDD containing the counted extracted urls.
    """
    AUTHOR_ID, SUBREDDIT_ID, URL_ID = 0, 1, 2
    assert count_on in ["all", "subreddit", "author"], "count_on must be one of ['all', 'subreddit', 'author']"
    if count_on == "all":
        urls_rdd = (rdd
                    .map(lambda x: (x[URL_ID], 1))  # (url, 1)
                    .reduceByKey(add) #Reduce the urls by key
                    .sortBy(lambda x: x[1], ascending=False)  # Sort by count
        )
    else:
        count_on_field = AUTHOR_ID if count_on == "author" else SUBREDDIT_ID
        urls_rdd = (rdd
                 .map(lambda x: ((x[count_on_field], x[URL_ID]), 1))  # ((subreddit or author, url), 1)
                 .reduceByKey(add)  # Reduce the urls by key
                 .map(lambda x: (x[0][0], (x[0][1], x[1])))  # (subreddit, (url, count))
                 .groupByKey()  # Group by subreddit
                 .mapValues(list)  # Convert to list
                 .mapValues(lambda x: sorted(x, key=lambda y: y[1], reverse=True))  # Sort by count
        )

    return urls_rdd

def get_youtube_info(rdd:RDD, count_on="all") -> RDD:
    """
    This function extracts the video ID from the URL.
    Args:
        rdd (RDD): The RDD containing the data.
    Returns:
        youtube_rdd (RDD): The RDD containing the video ID.
    """
    AUTHOR_ID, SUBREDDIT_ID, URL_ID = 0, 1, 2
    assert count_on in ["all", "subreddit", "author"], "count_on must be one of ['all', 'subreddit', 'author']"
    youtube_rdd = (rdd
                   .filter(lambda x: ("youtube.com" in x[URL_ID]) or ("youtu.be" in x[URL_ID])) #Filter the YouTube URLs
                   .map(lambda x: (x[AUTHOR_ID], x[SUBREDDIT_ID], __extract_video_id(x[URL_ID]))) #Extract the video ID
                   .filter(lambda x: x[2] is not None) #Filter the None values
                   )
    if count_on == "all":
        youtube_rdd = (youtube_rdd
                    .map(lambda x: (x[URL_ID], 1))  # (video_id, 1)
                    .reduceByKey(add) #Reduce the video ID by key
                    .sortBy(lambda x: x[1], ascending=False)  # Sort by count
        )
    else:
        count_on_field = AUTHOR_ID if count_on == "author" else SUBREDDIT_ID
        youtube_rdd = (youtube_rdd
                    .map(lambda x: ((x[count_on_field], x[2]), 1))  # ((subreddit or author, video_id), 1)
                    .reduceByKey(add)  # Reduce the video ID by key
                    .map(lambda x: (x[0][0], (x[0][1], x[1])))  # (subreddit, (video_id, count))
                    .groupByKey()  # Group by subreddit
                    .mapValues(list)  # Convert to list
                    .mapValues(lambda x: sorted(x, key=lambda y: y[1], reverse=True))  # Sort by count
        )
    return youtube_rdd

def __extract_video_id(url):
    """Extracts video ID from both standard and shortened YouTube URLs"""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def __extract_urls(text:str, url:str) -> List[str]:
    """
    Function that extracts URLs from a text.

    Args:
        text (str): Text to extract URLs from.
        url (str): URL to add to the text.
        
    Returns:
        domains (List[str]): List of urls extracted from the URLs.
    """
    #Here we extract the URLs from the text.
    url_pattern = r'https?://\S+' #Regular expression to match URLs
    urls = re.findall(url_pattern, text)
    #Here we parse the individual URL and add it to the list of URLs.
    urls.append(url)
    #We want only the unique URLs.
    urls = list(set(urls))

    return urls


def __get_subreddit_list() -> List[str]:
    """
    This function returns the list of popular subreddits that have demographic scores available.
    Args:
        None
    Returns:
        subreddit_scores (list): The list of popular subreddits.
    """
    DEMOGRAPHICS_SCORES_PATH = f"{DATA_PATH}/seed_communities.csv"
    subreddit_list = pd.read_csv(DEMOGRAPHICS_SCORES_PATH, header=0)["id"].tolist()
    return subreddit_list

def __get_bot_list() -> List[str]:
    """
    This function gets the bot list to remove them from the data.
    Args:
        None
    Returns:
        bots (list): The list of bots.
    """
    BOTS_PATH = f"{REDDIT_DATA_PATH}/new_bot_list.joblib"
    BOTS_PATH_2 = f"{REDDIT_DATA_PATH}/string_bot_list.joblib"
    bot_list = joblib.load(BOTS_PATH)
    bot_list_2 = joblib.load(BOTS_PATH_2)
    bots = list(set(bot_list).union(set(bot_list_2)))
    return bots

def __resilient_json(line:str) -> Dict[str, str]:
    """
    This function loads a JSON line and returns a dictionary.
    Args:
        line (str): The JSON line.
    Returns:
        json.loads(line) (dict): The dictionary.
    """
    try:
        return json.loads(line)
    except:
        return {}
    
def __hash_filter(s:str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100

def safe_urlparse(url):
    try:
        parsed = tldextract.extract(url)
        domain = parsed.domain
        suffix = parsed.suffix
        if domain == "" or suffix == "":
            return None
        return domain + "." + suffix
    except:
        return None