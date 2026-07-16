from sklearn.preprocessing import QuantileTransformer
from dotenv import load_dotenv
from typing import List, Dict
import pandas as pd
import praw
import time
import os

DATA_PATH = "data_shared/supplementary_data/"
REDDIT_DATA_PATH = "data_shared/reddit/"

def get_subscriber_count(reddit_instance: praw.Reddit, subreddit_names: List[str]) -> Dict[str, int]:
    """
    Get the subscriber count for a list of subreddits

    Args:
    reddit_instance: A praw.Reddit instance
    subreddit_names: A list of subreddit names

    Returns:
    A dictionary with subreddit names as keys and subscriber counts as values
    """

    #Here we create a dictionary to store the subscriber count for each subreddit
    subscribers = {}
    #Here we iterate over the list of subreddit names and get the subscriber count for each
    for subreddit_name in subreddit_names:
        #We stop for 2 seconds to avoid hitting the Reddit API too hard
        time.sleep(2)
        try:
            subreddit = reddit_instance.subreddit(subreddit_name)
            subscribers[subreddit_name] = subreddit.subscribers
        except Exception as e:
            subscribers[subreddit_name] = None

    return subscribers

def normalize_subscriber_count(subscribers_dict: Dict[str, int], min_value:float = 0.1, max_value:float = 1) -> Dict[str, float]:
    """
    Normalize the subscriber count for a list of subreddits

    Args:
    subscribers_dict: A dictionary with subreddit names as keys and subscriber counts as values
    min_value: The minimum value for normalization
    max_value: The maximum value for normalization

    Returns:
    A dictionary with subreddit names as keys and normalized subscriber counts as values
    """
    #Here we extract valid subscriber counts
    valid_subscribers = [count for count in subscribers_dict.values() if count is not None]
    #Here we compute min and max for normalization
    min_subscribers = min(valid_subscribers)
    max_subscribers = max(valid_subscribers)

    #Now, we normalize valid values to [min_value, max_value]
    normalized_values = []
    normalized_subscribers = {}
    for subreddit, count in subscribers_dict.items():
        if count is not None:
            #Normalize using the adjusted range
            normalized_value = min_value + ((count - min_subscribers) * (max_value - min_value) / (max_subscribers - min_subscribers))
            normalized_subscribers[subreddit] = normalized_value
            normalized_values.append(normalized_value)
        else:
            normalized_subscribers[subreddit] = min_value

    return normalized_subscribers

def robust_normalizer(subscribers_dict: Dict[str, int]) -> Dict[str, float]:
    """
    Normalize the subscriber count for a list of subreddits by using a quantile transformer

    Args:
    subscribers_dict: A dictionary with subreddit names as keys and subscriber counts as values

    Returns:
    A dictionary with subreddit names as keys and normalized subscriber counts as values
    """

    #Here we extract valid subscriber counts
    valid_subscribers = [count for count in subscribers_dict.values() if count is not None]
    #Here we reshape the data
    valid_subscribers = [[count] for count in valid_subscribers]

    #Here we create the quantile transformer
    transformer = QuantileTransformer(output_distribution='uniform')
    #Here we fit the transformer
    transformer.fit(valid_subscribers)

    #Now we normalize the subscriber counts
    normalized_subscribers = {}
    for subreddit, count in subscribers_dict.items():
        if count is not None:
            normalized_subscribers[subreddit] = transformer.transform([[count]])[0][0]
        else:
            normalized_subscribers[subreddit] = 0.1

    return normalized_subscribers

if __name__ == "__main__":
    #First we load the environment variables from the .env file
    load_dotenv()

    #Here we create a Reddit instance using the praw library
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_USER_ID"),   
        client_secret=os.getenv("REDDIT_SECRET"),
        user_agent=os.getenv("REDDIT_AGENT")
    )

    #Here we read the network
    df = pd.read_csv(f"{DATA_PATH}/filtered_attention_flow_graph_0.01_sampled.tsv", sep='\t', names=['source', 'target', 'weight'])
    #Here we read the communities file
    communities = pd.read_csv(f"{REDDIT_DATA_PATH}/communities_infomap_0.01.csv", header=0)
    
    #Here we define a list of community seed subreddits we want to analyze
    seeds = ["conspiracy", "spirituality", "Conservative", "environment", "socialism"]

    #We obtain the subreddits that belong to the same community as the seeds
    community_ids = communities[communities.id.isin(seeds)].community.values
    subreddit_names = communities[communities.community.isin(community_ids)].id.values

    #We get the subscriber count for the subreddits
    print("Getting subscriber count for all subreddits...")
    subscribers = get_subscriber_count(reddit, subreddit_names)
    #We normalize the subscriber count
    print("Normalizing subscriber count...")
    normalized_subscribers = normalize_subscriber_count(subscribers)
    #We apply the robust normalizer
    print("Applying robust normalizer...")
    normalized_subscribers = robust_normalizer(subscribers)

    #We save the community dataframe with only the obtained subreddits and adding the raw and normalized subscriber count
    communities = communities[communities.id.isin(subreddit_names)]
    communities['subscribers'] = communities.id.map(subscribers)
    communities['normalized_subscribers'] = communities.id.map(normalized_subscribers)
    communities['robust_normalized_subscribers'] = communities.id.map(normalized_subscribers)

    #We save the communities dataframe
    communities.to_csv(f"{DATA_PATH}/seed_communities.csv", index=False)
    print("Community data saved!")
