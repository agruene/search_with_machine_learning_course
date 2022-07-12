#!/usr/bin/env python
# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
from getpass import getpass
from urllib.parse import urljoin
import pandas as pd
import fileinput
import logging
import fasttext
import re
import pprint as pp
import sys 
import copy

DEFAULT_MIN_CATEGORIES_PROBABILITY = 0.5
DEFAULT_USE_MULTIPLE_CATEGORIES = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s')

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, click_prior_query, filters, sort="_score", sortDir="desc", size=10, source=None, categories=None):
    query_filters = filters
    if categories:
        category_filter = {
            "terms": {
                "categoryPathIds.keyword" :categories
            }
        }
        if query_filters:
            query_filters.append(category_filter)
        else:
            query_filters = [category_filter]
    
    query_obj = {
        "size": size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [

                        ],
                        "should": [ 
                            {
                                "match": {
                                    "name": {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": ["name^10", "name.hyphens^10", "shortDescription^5",
                                                "longDescription^5", "department^0.5", "sku", "manufacturer", "features",
                                                "categoryPath"]
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": query_filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]
            }
        }
    }
    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj

def normalize_query(query: str) -> str:
    normalized_query = query.lower()
    normalized_query = re.sub("[^a-z0-9]" , " ", normalized_query)
    normalized_query = re.sub("\s+" , " ", normalized_query)
    return normalized_query

def categorize_query(user_query: str, min_categories_probability: float = DEFAULT_MIN_CATEGORIES_PROBABILITY, use_multiple_categories: bool = DEFAULT_USE_MULTIPLE_CATEGORIES):
    categorization_model = fasttext.load_model("/workspace/models/query_classifier_minq10000.bin")
    normalized_query = normalize_query(query=user_query)
    result = [] # list of categories
    summed_probabilities = 0.0
    if not use_multiple_categories:  # just one category in output desired
        categories, probabilities = categorization_model.predict(normalized_query)
        if (categories is not None) and (len(categories)>0):
            category_label = categories[0]
            category = category_label[len("__label__"):]
            probability = probabilities[0]
            if probability > min_categories_probability:
                result = [category]
                summed_probabilities += probability
    else:  # try using more than 1 category for output
        predictions = categorization_model.predict(normalized_query, k=10)
        pp.pprint(predictions)
        categories = []
        probabilities = []
        if (predictions is not None) and (len(predictions) > 0):
            categories = predictions[0]
            probabilities = predictions[1]
        summed_probabilities = 0.0
        result = []
        for i in range(0, len(categories)):
            category_label = categories[i]
            probability = probabilities[i]
            category = category_label[len("__label__"):]
            result.append(category)
            summed_probabilities += probability
            if summed_probabilities > min_categories_probability:
                break
    print("user query: '{}' -> normalized query: '{}' -> predicted categories: {}, (summed) probability: {}, min_probability: {}". format(
        user_query, normalized_query, result, summed_probabilities, min_categories_probability
    ))
    return result

def search(client, user_query, index="bbuy_products", sort="_score", sortDir="desc", 
    min_categories_probability=DEFAULT_MIN_CATEGORIES_PROBABILITY, use_multiple_categories=DEFAULT_USE_MULTIPLE_CATEGORIES):
    categories = categorize_query(user_query=user_query, min_categories_probability=min_categories_probability, use_multiple_categories=use_multiple_categories)
    query_obj = create_query(user_query, click_prior_query=None, filters=None, sort=sort, sortDir=sortDir,
                             source=["name", "shortDescription", "categoryPathIds"], categories=categories)
    logging.info(query_obj)
    count_obj = {}
    count_obj["query"] = query_obj["query"]
    count_response = client.count(count_obj, index=index)
#    pp.pprint(count_response) # debugging
#    exit(0)
    total_count = int(count_response["count"])
    print("total results: {}".format(total_count))
    response = client.search(query_obj, index=index)
#    pp.pprint(response) # debugging
#    exit(0)
    if response and response['hits']['hits'] and len(response['hits']['hits']) > 0:
        # - nicer output of results including categories -
        hits = response['hits']['hits']
        print("showing: {} results".format(len(hits)))
        i=0
        for hit in hits:
            i += 1
            print("{}. id: {}, name: '{}', categories: {}".format(i, hit["_id"], hit["_source"]["name"][0], hit["_source"]["categoryPathIds"])) 
 #       print(json.dumps(response, indent=2))


if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")
    general.add_argument("-i", "--index", default="bbuy_products",
                         help="The name of the main index to search")
    general.add_argument("-s", "--host", default="localhost",
                         help="The OpenSearch host name")
    general.add_argument("-p", "--port", type=int, default=9200,
                         help="The OpenSearch port")
    general.add_argument("--user",
                         help="The OpenSearch admin.  If this is set, the program will prompt for password too. If not set, use default of admin/admin")
    general.add_argument("--min_categories_probability", type=float, default=1.0,
                         help="The minimum prediction probability that all used query categories summed together must reach. If not provided, categories are not used.")
    general.add_argument("--use_multiple_categories", default=False, action="store_true")
    
    args = parser.parse_args()
    args, unknownargs = parser.parse_known_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    host = args.host
    port = args.port
    if args.user:
        password = getpass()
        auth = (args.user, password)
    min_categories_probability = args.min_categories_probability
    use_multiple_categories = args.use_multiple_categories
    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    index_name = args.index
    query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"
    print(query_prompt)
    for line in sys.stdin:
 #   for line in fileinput.input():
        query = line.rstrip()
        if query.lower() == "exit":
            break
        search(client=opensearch, user_query=query, index=index_name, min_categories_probability=min_categories_probability, use_multiple_categories=use_multiple_categories)
        print(query_prompt)
