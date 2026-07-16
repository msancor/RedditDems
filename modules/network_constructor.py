from typing import List, Tuple, Dict
from operator import add 
from pyspark import RDD
import pandas as pd
import hashlib
import joblib
import json
import ast

DATA_PATH = "data_shared/reddit/"


def calculate_attention_flow_graph(rescaled_weights:RDD, total_months:int) -> bool:
    """
    This function calculates the attention flow graph by averaging the attention flow scores.
    Args:
        aggregate_by_user (RDD): The RDD containing the aggregated attention flow by subreddit pair and timestep.
        total_months (int): The total number of months.
    Returns:
        subreddit_pair_scores (RDD): The RDD containing the attention flow graph by averaging the attention flow scores.
    """
    SUBREDDIT_I_ID, SUBREDDIT_J_ID, GEOMETRIC_SCALED_WEIGHT = 0, 1, -1
    KEY_ID, VALUE_ID = 0, 1
    
    subreddit_pair_scores = (
        rescaled_weights
        .map(lambda x: ((x[SUBREDDIT_I_ID], x[SUBREDDIT_J_ID]), x[GEOMETRIC_SCALED_WEIGHT]))  # ((subreddit_i, subreddit_j), score)
        .reduceByKey(add)  # sum scores
        .map(lambda x: (x[KEY_ID][SUBREDDIT_I_ID], x[KEY_ID][SUBREDDIT_J_ID], x[VALUE_ID] /total_months))  # (subreddit_i, subreddit_j, avg_score)
    )

    subreddit_pair_scores.map(__rdd_line_to_string).saveAsTextFile(f"{DATA_PATH}/attention_flow_graph_sampled")

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
    aggregate_by_user = __aggregated_attention_flow(attention_flow)
    rescaled_weights = __weight_rescaling(aggregate_by_user)
    rescaled_weights.saveAsTextFile(f"{DATA_PATH}/temporal_graph_data_sampled", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
    return rescaled_weights

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
    subreddit_counts = (
        rdd
        .map(__resilient_json) #Load the JSON
        .map(lambda x: (x.get("author"), x.get("subreddit"), x.get("score"))) #Get the necessary fields
        .filter(lambda x: __hash_filter(x[AUTHOR_ID]) < percentage) #Sample the users
        .filter(lambda x: x[AUTHOR_ID] not in bots) #Filter the bots
        .filter(lambda x: x[SCORE_ID] > 1) #Filter the submissions with score greater than 1
        .filter(lambda x: x[SUBREDDIT_ID] in subreddit_list) #Filter the subreddits with demographic scores available
        .map(lambda x: ((x[AUTHOR_ID], x[SUBREDDIT_ID]), 1))  #((author, subreddit), 1)
        .reduceByKey(add) #Count the number of submissions per author and subreddit
        )
    
    return subreddit_counts

def data_preprocessing_bis(rdd:RDD, percentage=10) -> RDD:
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
        author_counts (RDD): The RDD containing the number of submissions per author and timestep.
    """
    AUTHOR_ID, SUBREDDIT_ID, SCORE_ID = 0, 1, 2
    subreddit_list = __get_subreddit_list()
    bots = __get_bot_list()
    author_counts = (
        rdd
        .map(__resilient_json) #Load the JSON
        .map(lambda x: (x.get("author"), x.get("subreddit"), x.get("score"))) #Get the necessary fields
        .filter(lambda x: __hash_filter(x[AUTHOR_ID]) < percentage) #Sample the users
        .filter(lambda x: x[AUTHOR_ID] not in bots) #Filter the bots
        .filter(lambda x: x[SCORE_ID] > 1) #Filter the submissions with score greater than 1
        .filter(lambda x: x[SUBREDDIT_ID] in subreddit_list) #Filter the subreddits with demographic scores available
        .map(lambda x: (x[AUTHOR_ID], 1))  #(author, 1)
        .reduceByKey(add) #Count the number of submissions per author
        )
    
    return author_counts

def get_attention_ratio(subreddit_counts:RDD, timestep:int) -> RDD:
    """
    This function calculates the fraction of submissions per author, subreddit and timestep.
    Args:
        subreddit_counts (RDD): The RDD containing the number of submissions per author, subreddit and timestep.
        timestep (int): The timestep.
    Returns:
        fraction_of_submissions (RDD): The RDD containing the fraction of submissions per author, subreddit and timestep.
    """
    AUTHOR_ID, SUBREDDIT_ID = 0, 1
    KEY_ID, COUNT_ID = 0, 1
    
    author_timestep_totals = (
        subreddit_counts
        .map(lambda x: (x[KEY_ID][AUTHOR_ID], x[COUNT_ID]))  # (author, count)
        .reduceByKey(add)  # total submissions for each author at a timestep (author, total_count)
        )
    
    fraction_of_submissions = (
        subreddit_counts
        .map(lambda x: (x[KEY_ID][AUTHOR_ID], (x[KEY_ID][SUBREDDIT_ID], x[COUNT_ID])))  # (author, (subreddit, count))
        .join(author_timestep_totals)  # Join to get (author, ((subreddit, count), total_count))
        .map(lambda x: (x[AUTHOR_ID], x[SUBREDDIT_ID][KEY_ID][KEY_ID], timestep, x[SUBREDDIT_ID][KEY_ID][COUNT_ID]/x[SUBREDDIT_ID][COUNT_ID]))  # (author, subreddit, timestep, fraction)
        )

    return fraction_of_submissions

def __hash_filter(s:str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100

def __get_subreddit_list() -> List[str]:
    """
    This function returns the list of popular subreddits that have demographic scores available.
    Args:
        None
    Returns:
        subreddit_scores (list): The list of popular subreddits.
    """
    DEMOGRAPHICS_SCORES_PATH = f"{DATA_PATH}/subreddit_list.joblib"
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
    BOTS_PATH = f"{DATA_PATH}/new_bot_list.joblib"
    BOTS_PATH_2 = f"{DATA_PATH}/string_bot_list.joblib"
    bot_list = joblib.load(BOTS_PATH)
    bot_list_2 = joblib.load(BOTS_PATH_2)
    bots = list(set(bot_list).union(set(bot_list_2)))
    return bots

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

def __rdd_line_to_string(x: Tuple[str, str, float]) -> str:
    """
    This function converts a tuple to a string.
    Args:
        x (tuple): The tuple.
    Returns:
        (str): The string.
    """
    s_i, s_j, weight = x # Unpack the tuple
    return s_i + "," + s_j + "," + str(weight)
    
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
    
def __aggregated_attention_flow(attention_flow:RDD) -> RDD:
    """
    This function aggregates the attention flow by subreddit pair and timestep.
    Args:
        attention_flow (RDD): The RDD containing the attention flow per author, subreddit and timestep.
    Returns:
        aggregate_by_user (RDD): The RDD containing the aggregated attention flow by subreddit pair and timestep.
    """
    KEY_ID, VALUE_ID = 0, 1
    AUTHOR_ID, TIME_ID = 0, 1
    
    aggregate_by_user = (
        attention_flow
        .flatMap(lambda x: [((subreddit_i, subreddit_j, x[KEY_ID][TIME_ID]), att_score) for subreddit_i, subreddit_j, att_score in x[VALUE_ID]])
        .reduceByKey(add) # Aggregate the outer products by (subreddit_i, subreddit_j, timestep)
    )

    return aggregate_by_user

def __weight_rescaling(aggregated_attention_flow:RDD) -> RDD:
    """
    This function rescales the weights of the aggregated attention flow by two methods:
    - Arithmetic Mean of the nodes in-strength and out-strength.
    - Geometric Mean of the nodes in-strength and out-strength.

    Args:
        aggregated_attention_flow (RDD): The RDD containing the aggregated attention flow by subreddit pair and timestep.
    Returns:
        rescaled_weights (RDD): The RDD containing the rescaled weights of the aggregated attention flow.
    """
    KEY_ID, VALUE_ID = 0, 1
    SUBREDDIT_I_ID, SUBREDDIT_J_ID, TIME_ID = 0, 1, 2

    #Calculate the in-strength and out-strength of each node
    in_strength = (
        aggregated_attention_flow
        .map(lambda x: ((x[KEY_ID][SUBREDDIT_J_ID], x[KEY_ID][TIME_ID]), x[VALUE_ID]))  # ((subreddit_j, timestep), weight)
        .reduceByKey(add)  # sum weights
    )
    out_strength = (
        aggregated_attention_flow
        .map(lambda x: ((x[KEY_ID][SUBREDDIT_I_ID], x[KEY_ID][TIME_ID]), x[VALUE_ID]))  # ((subreddit_i, timestep), weight)
        .reduceByKey(add)  # sum weights
    )

    #Join the in-strength and out-strength RDDs with the aggregated attention flow RDD
    joined_strengths = (
        aggregated_attention_flow
        .map(lambda x: ((x[KEY_ID][SUBREDDIT_J_ID], x[KEY_ID][TIME_ID]), (x[KEY_ID][SUBREDDIT_I_ID], x[VALUE_ID])))  # ((subreddit_j, timestep), (subreddit_i, weight))
        .join(in_strength)  # ((subreddit_j, timestep), ((subreddit_i, weight), in_strength))
        .map(lambda x: ((x[1][0][0], x[0][1]), (x[0][0], x[1][0][1], x[1][1])))  # ((subreddit_i, timestep), (subreddit_j, weight, in_strength))
        .join(out_strength)  # ((subreddit_i, timestep), ((subreddit_j, weight, in_strength), out_strength))
        .map(lambda x: (x[0][0], x[1][0][0], x[0][1], x[1][0][1], x[1][0][2], x[1][1]))  # (subreddit_i, subreddit_j, timestep, weight, in_strength, out_strength)
    )

    #Calculate the arithmetic mean of the in-strength and out-strength
    rescaled_weights = (
        joined_strengths
        .map(lambda x: (x[0], x[1], x[2], x[3], __rescale_weight(x[3], x[4], x[5], "arithmetic"), __rescale_weight(x[3], x[4], x[5], "geometric")))  # (subreddit_i, subreddit_j, timestep, weight, arithmetic_rescaled_weight, geometric_rescaled_weight)
    )

    return rescaled_weights

def __rescale_weight(original_weight:float, in_strength:float, out_strength:float, technique:str) -> float:
    """
    This function rescales the weight of the attention flow.
    Args:
        original_weight (float): The original weight.
        in_strength (float): The in-strength of the node.
        out_strength (float): The out-strength of the node.
        technique (str): The rescaling technique.
    Returns:
        (float): The rescaled weight.
    """
    if technique == "arithmetic":
        rescaling_factor = (in_strength + out_strength)/2
        return original_weight/rescaling_factor
    elif technique == "geometric":
        rescaling_factor = (in_strength*out_strength)**0.5
        return original_weight/rescaling_factor
    else:
        return original_weight
    
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
