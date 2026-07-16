from typing import List, Tuple, Dict
from operator import add 
from pyspark import RDD
import pandas as pd
import hashlib
import joblib
import json
import ast

REDDIT_DATA_PATH = "data_shared/reddit/"
MAIN_DATA_PATH = "data_shared/main_data/"
SUPPLEMENTARY_DATA_PATH = "data_shared/supplementary_data/"


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
        subreddit_counts (RDD): The RDD containing the number of submissions per author, subreddit and timestep.
    """
    AUTHOR_ID, SUBREDDIT_ID, SCORE_ID = 0, 1, 2
    subreddit_list = __get_subreddit_list()
    bots = __get_bot_list()
    processed_rdd = (
        rdd
        .map(__resilient_json) #Load the JSON
        .map(lambda x: (x.get("author"), x.get("subreddit"), x.get("score"))) #Get the necessary fields
        .filter(lambda x: __hash_filter(x[AUTHOR_ID]) < percentage) #Sample the users
        .filter(lambda x: x[AUTHOR_ID] not in bots) #Filter the bots
        .filter(lambda x: x[SCORE_ID] > 1) #Filter the submissions with score greater than 1
        .filter(lambda x: x[SUBREDDIT_ID] in subreddit_list) #Filter the subreddits with demographic scores available
        )
    
    return processed_rdd

def calculate_attention_flow_graph(aggregate_by_user:RDD, total_months:int, propagate_scores:bool=True) -> bool:
    """
    This function calculates the attention flow graph by averaging the attention flow scores.
    Args:
        aggregate_by_user (RDD): The RDD containing the attention flow by subreddit pair and timestep.
        total_months (int): The total number of months.
    Returns:
        subreddit_pair_scores (RDD): The RDD containing the attention flow graph by averaging the attention flow scores.
    """
    SUBREDDIT_I_ID, SUBREDDIT_J_ID, WEIGHTS = 0, 1, -1
    KEY_ID, VALUE_ID = 0, 1

    filtered_edges = __filter_edges()
    
    subreddit_pair_scores = (
        aggregate_by_user
        .map(lambda x: ((x[KEY_ID][SUBREDDIT_I_ID], x[KEY_ID][SUBREDDIT_J_ID]), x[WEIGHTS]))  # ((subreddit_i, subreddit_j), (age_score, ...))
        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3]))  # ((subreddit_i, subreddit_j), (age_score, ...))
        .map(lambda x: (x[KEY_ID][SUBREDDIT_I_ID], x[KEY_ID][SUBREDDIT_J_ID], x[VALUE_ID][0]/total_months, x[VALUE_ID][1]/total_months, x[VALUE_ID][2]/total_months, x[VALUE_ID][3]/total_months))  # (subreddit_i, subreddit_j, age_score, ...)
        .filter(lambda x: (x[SUBREDDIT_I_ID], x[SUBREDDIT_J_ID]) in filtered_edges)
    )
    if propagate_scores:
        subreddit_pair_scores.map(__att_flow_to_string).saveAsTextFile(f"{REDDIT_DATA_PATH}/attention_flow_graph_scores")
    else:
        subreddit_pair_scores.map(__att_flow_to_string).saveAsTextFile(f"{REDDIT_DATA_PATH}/attention_flow_graph_scores_sparse")

    return True

def calculate_temporal_graph_data(full_rdd:RDD) -> RDD:
    """
    This function calculates the raw graph data.
    Args:
        full_rdd (RDD): The RDD containing the data.
        total_months (int): The total number of months.
    Returns:
        (bool): True if the function runs successfully.
    """
    fraction_of_submissions = full_rdd.map(__resilient_eval)
    timestep_differences = __get_attention_delta(fraction_of_submissions)
    attention_flow = __get_attention_flow(timestep_differences)
    return attention_flow

def aggregated_attention_flow(attention_flow:RDD, propagate_scores:bool=True) -> RDD:
    """
    This function aggregates the attention flow by subreddit pair and timestep.
    Args:
        attention_flow (RDD): The RDD containing the attention flow per author, subreddit and timestep.
    Returns:
        aggregate_by_user (RDD): The RDD containing the aggregated attention flow by subreddit pair and timestep.
    """
    KEY_ID, VALUE_ID = 0, 1
    AUTHOR_ID, TIME_ID = 0, 1
    age_scores = __get_user_scores("age", propagate_scores)
    gender_scores = __get_user_scores("gender", propagate_scores)
    affluence_scores = __get_user_scores("affluence", propagate_scores)
    partisan_scores = __get_user_scores("partisan", propagate_scores)
    
    aggregate_by_user = (
        attention_flow
        .flatMap(lambda x: [((subreddit_i, subreddit_j, x[KEY_ID][TIME_ID]),
                             (att_score*age_scores.get(x[KEY_ID][AUTHOR_ID], 0), att_score*gender_scores.get(x[KEY_ID][AUTHOR_ID], 0),
                              att_score*affluence_scores.get(x[KEY_ID][AUTHOR_ID], 0),
                              att_score*partisan_scores.get(x[KEY_ID][AUTHOR_ID], 0))) for subreddit_i, subreddit_j, att_score in x[VALUE_ID]])
        # Aggregate the outer products by (subreddit_i, subreddit_j, timestep) for each demographic
        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3]))  # ((subreddit_i, subreddit_j, timestep), 
    )

    return aggregate_by_user


def calculate_user_scores(processed_rdd:RDD) -> bool:
    """
    This function calculates the demographic scores per author.
    Args:
        processed_rdd (RDD): The RDD containing the data.
    Returns:
        (bool): True if the function runs successfully.
    """
    AUTHOR_ID, SUBREDDIT_ID, SCORE_ID = 0, 1, 2
    
    number_of_submissions_user_subreddit = (
        processed_rdd
        .map(lambda x: ((x[AUTHOR_ID], x[SUBREDDIT_ID]), 1)) #((author, subreddit), 1)
        .reduceByKey(add) #((author, subreddit), count)
    )
    #We obtain \sum_s N_{u,s} where u is the author, s is the subreddit, N_{u,s} is the number of submissions of author u to subreddit s
    subreddit_counts = (
        number_of_submissions_user_subreddit
        .map(lambda x: (x[0][AUTHOR_ID], x[1])) #(author, count)
        .reduceByKey(add) #(author, count)
    )

    #We obtain the user demographic scores with score propagation and without
    demographic_scores = __get_user_demo_scores(number_of_submissions_user_subreddit, True)
    sparse_demographic_scores = __get_user_demo_scores(number_of_submissions_user_subreddit, False)

    #Finally we normalize the scores by the subreddit counts
    normalized_demographic_scores = (
        demographic_scores
        .join(subreddit_counts) #(author, ((\sum_s N_{u,s} s_s, ...), count))
        .map(lambda x: (x[0], x[1][0][0]/x[1][1], x[1][0][1]/x[1][1], x[1][0][2]/x[1][1], x[1][0][3]/x[1][1]))
    )

    normalized_sparse_demographic_scores = (
        sparse_demographic_scores
        .join(subreddit_counts) #(author, ((\sum_s N_{u,s} s_s, ...), count))
        .map(lambda x: (x[0], x[1][0][0]/x[1][1], x[1][0][1]/x[1][1], x[1][0][2]/x[1][1], x[1][0][3]/x[1][1]))
    )

    #We save the normalized demographic scores
    normalized_demographic_scores.map(__rdd_line_to_string).saveAsTextFile(f"{REDDIT_DATA_PATH}/normalized_demographic_scores")
    normalized_sparse_demographic_scores.map(__rdd_line_to_string).saveAsTextFile(f"{REDDIT_DATA_PATH}/normalized_sparse_demographic_scores")

    return True

def __get_user_demo_scores(number_of_submissions_user_subreddit:RDD, propagate_scores=True) -> RDD:
    
    AUTHOR_ID, SUBREDDIT_ID, SCORE_ID = 0, 1, 2

    #Here we load the demographic scores for each demographic axis
    age_scores = __get_demographic_scores("age", propagate_scores)
    gender_scores = __get_demographic_scores("gender", propagate_scores)
    affluence_scores = __get_demographic_scores("affluence", propagate_scores)
    partisan_scores = __get_demographic_scores("partisan", propagate_scores)

    #We obtain \frac{\sum_s N_{u,s} s_s} where s_s is the demographic score of subreddit s for a given demographic
    #In this case we are taking into account: age, gender, affluence, and partisan
    demographic_scores = (
        number_of_submissions_user_subreddit
        .map(lambda x: (x[0][AUTHOR_ID], (x[1]*age_scores.get(x[0][SUBREDDIT_ID], 0),
                                          x[1]*gender_scores.get(x[0][SUBREDDIT_ID], 0), x[1]*affluence_scores.get(x[0][SUBREDDIT_ID], 0),
                                          x[1]*partisan_scores.get(x[0][SUBREDDIT_ID], 0))))
        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3])) #(author, (\sum_s N_{u,s} s_s, ...))
    )
    return demographic_scores

def __get_attention_flow(timestep_differences:RDD) -> RDD:
    """
    This function calculates the attention flow per author, subreddit and timestep.
    Args:
        timestep_differences (RDD): The RDD containing the delta of the fraction of submissions per author, subreddit and timestep.
    Returns:
        rdd_outer_product (RDD): The RDD containing the attention flow per author, subreddit and timestep.
    """
    AUTHOR_ID, SUBREDDIT_ID, TIME_ID, DELTA_ID = 0, 1, 2, 3
    KEY_ID, VALUE_ID = 0, 1
    positive_deltas = (
        timestep_differences
        .filter(lambda x: x[DELTA_ID] > 0)  # filter out negative deltas
        .map(lambda x: ((x[AUTHOR_ID], x[TIME_ID]), (x[SUBREDDIT_ID],x[DELTA_ID])))  # ((author, timestep), (subreddit, delta))
        .groupByKey().mapValues(list)  # Group by author and timestep and collect a list of (subreddit, delta) tuples
    )
    negative_deltas = (
        timestep_differences
        .filter(lambda x: x[DELTA_ID] < 0)  # filter out positive deltas
        .map(lambda x: ((x[AUTHOR_ID], x[TIME_ID]), (x[SUBREDDIT_ID],x[DELTA_ID])))  # ((author, timestep), (subreddit, delta))
        .groupByKey().mapValues(list)  # Group by author and timestep and collect a list of (subreddit, delta) tuples
    )
    POS_DELTA_ID, NEG_DELTA_ID = 0, 1
    rdd_outer_product = (
        positive_deltas
        .join(negative_deltas)  # ((author, timestep), ([(subreddit, delta), ...], [(subreddit, delta), ...]))
        .map(lambda x: (x[KEY_ID], __outer_product(x[VALUE_ID][POS_DELTA_ID], x[VALUE_ID][NEG_DELTA_ID])))  # ((author, timestep), [(subreddit_i, subreddit_j, outer_product),...])
    )
    return rdd_outer_product

def __get_attention_delta(fraction_of_submissions:RDD) -> RDD:
    """
    This function calculates the delta of the fraction of submissions per author, subreddit and timestep.
    Args:
        fraction_of_submissions (RDD): The RDD containing the fraction of submissions per author, subreddit and timestep.
    Returns:
        timestep_differences (RDD): The RDD containing the delta of the fraction of submissions per author, subreddit and timestep.
    """
    AUTHOR_ID, SUBREDDIT_ID, TIME_ID, FRACTION_ID = 0, 1, 2, 3
    # RDD for next timestep t+1 (shifted by 1 timestep)
    t_next = fraction_of_submissions.map(lambda x: ((x[AUTHOR_ID], x[SUBREDDIT_ID], x[TIME_ID]-1), x[FRACTION_ID]))  # ((author, subreddit, timestep-1), value)

    KEY_ID, VALUE_ID = 0, 1
    FRACTION_T_ID, FRACTION_T1_ID = 0, 1
    timestep_differences = (
        fraction_of_submissions
        .map(lambda x: ((x[AUTHOR_ID], x[SUBREDDIT_ID], x[TIME_ID]), x[FRACTION_ID]))  # ((author, subreddit, timestep), fraction)
        .fullOuterJoin(t_next)  # ((author, subreddit, timestep), (fraction_t, fraction_t+1))
        .filter(lambda x: (x[VALUE_ID][FRACTION_T_ID] is None) or (x[VALUE_ID][FRACTION_T1_ID] is None))  # filter out timesteps where we have both t and t+1
        .map(lambda x: (x[KEY_ID][AUTHOR_ID], x[KEY_ID][SUBREDDIT_ID], x[KEY_ID][TIME_ID], __calculate_delta(x[VALUE_ID][FRACTION_T_ID], x[VALUE_ID][FRACTION_T1_ID])))  # (author, subreddit, timestep, delta)
    )

    return timestep_differences

def __resilient_eval(line:str) -> Tuple[str, str, int, float]:
    """
    This function evaluates a line.
    Args:
        line (str): The line.
    Returns
        ast.literal_eval(line) (tuple): The tuple.
    """
    try:
        return ast.literal_eval(line)
    except:
        return ()
    
def __get_demographic_scores(demographic:str, propagate_scores:bool=True) -> Dict[str, float]:
    """
    This function loads the demographic scores.
    Args:
        demographic (str): The demographic of interest.
        propagate_scores (bool): Whether to propagate the scores or not.
    Returns:
        demographic_scores (dict): The demographic scores.
    """
    #We read the demographic scores from a CSV file
    if propagate_scores:
        demographic_scores = pd.read_csv(f"{SUPPLEMENTARY_DATA_PATH}/demographic_scores.csv")
    else:
        demographic_scores = pd.read_csv(f"{REDDIT_DATA_PATH}/demographic_scores_sparse.csv")
    #We convert the demographic scores to a dictionary with keys as subreddits and values as demographic scores
    demographic_scores = demographic_scores.set_index("community")[str(demographic)].to_dict()
    return demographic_scores


def __get_user_scores(demographic:str, propagate_scores:bool=True) -> Dict[str, float]:
    """
    This function loads the user scores.
    Args:
        demographic (str): The demographic of interest.
        propagate_scores (bool): Whether to propagate the scores
    Returns:
        user_demographic_scores (dict): The user scores.
    """
    #We read the user scores from a CSV file
    FILE_PATH = f"{REDDIT_DATA_PATH}/normalized_demographic_scores.tsv" if propagate_scores else f"{REDDIT_DATA_PATH}/normalized_sparse_demographic_scores.tsv"
    user_demographic_scores = pd.read_csv(FILE_PATH, sep=",", names=["user", "age", "gender", "affluence", "partisan"])
    #We convert the user scores to a dictionary with keys as authors and values as demographic scores
    user_demographic_scores = user_demographic_scores.set_index("user")[str(demographic)].to_dict()
    return user_demographic_scores

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
    
def __calculate_delta(value1:float, value2:float) -> float:
    """
    This function calculates the delta between two values.
    Args:
        value1 (float): The first value.
        value2 (float): The second value.
    Returns:
        (float): The delta between the two values.
    """
    if value1 is None:
        return value2
    return (-1)*value1

def __outer_product(positive_deltas:List[Tuple[str, float]], negative_deltas:List[Tuple[str, float]]) -> List[Tuple[str, str, float]]:
    """
    This function calculates the outer product of two lists of deltas.
    Args:
        positive_deltas (list): The list of positive deltas.
        negative_deltas (list): The list of negative deltas.
    Returns:
        (list): The list of outer products.
    """
    SUBREDDIT_ID, DELTA_ID = 0, 1
    outer_product = []
    neg_sum = sum([neg[DELTA_ID] for neg in negative_deltas])
    for pos in positive_deltas:
        for neg in negative_deltas:
            outer_product.append((pos[SUBREDDIT_ID], neg[SUBREDDIT_ID], pos[DELTA_ID]*neg[DELTA_ID]/neg_sum))
    return outer_product

def __rdd_line_to_string(x: Tuple[str, float, float, float, float]) -> str:
    """
    This function converts a tuple to a string.
    Args:
        x (tuple): The tuple.
    Returns:
        (str): The string.
    """
    author, s_1, s_2, s_3, s_4 = x # Unpack the tuple
    return author + "," + str(s_1) + "," + str(s_2) + "," + str(s_3) + "," + str(s_4)

def __att_flow_to_string(x: Tuple[str, str, float, float, float, float]) -> str:
    """
    This function converts a tuple to a string.
    Args:
        x (tuple): The tuple.
    Returns:
        (str): The string.
    """
    subreddit_i, subreddit_j, s_1, s_2, s_3, s_4 = x # Unpack the tuple
    return subreddit_i + "," + subreddit_j + "," + str(s_1) + "," + str(s_2) + "," + str(s_3) + "," + str(s_4)

def __get_subreddit_list() -> List[str]:
    """
    This function returns the list of popular subreddits that have demographic scores available.
    Args:
        None
    Returns:
        subreddit_scores (list): The list of popular subreddits.
    """
    DEMOGRAPHICS_SCORES_PATH = f"{REDDIT_DATA_PATH}/subreddit_list.joblib"
    subreddit_list = joblib.load(DEMOGRAPHICS_SCORES_PATH)
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

def __hash_filter(s:str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100

def __filter_edges()->List[Tuple[str, str]]:
    """
    This function filters the edges.
    Args:
        None
    Returns:
        edges (list): The list of edges.
    """
    filtered_graph = pd.read_csv(f"{SUPPLEMENTARY_DATA_PATH}/filtered_attention_flow_graph_0.01_sampled.tsv", sep="\t", names=["subreddit_i", "subreddit_j", "weight"])
    filtered_graph_tuples = list(zip(filtered_graph["subreddit_i"], filtered_graph["subreddit_j"]))
    return filtered_graph_tuples