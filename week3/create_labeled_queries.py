import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re
import pprint as pp
from collections import Counter
import copy

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

# --- functions ---
def check_category(category_to_check, current_parents_dict, current_category_count_dict):
    """prints out information about category_to_check using the provided current structures"""
    parent = "not existing"
    if category_to_check in current_parents_dict.keys():
        parent = current_parents_dict[category_to_check]
    print("The category '{}' has the parent '{}'.".format(category_to_check, parent))
    count = 0
    if category_to_check in current_category_count_dict.keys():
        count = current_category_count_dict[category_to_check]
    print("It has the count {}.".format(count))
    print("And it has the children {}.".format(", ".join([c for c, p in current_parents_dict.items() if p == category_to_check])))

# --- parsing the arguments ---
parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")
general.add_argument("--normalize", default=False, help="normalize the queries", action='store_true')

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)
do_normalize = args.normalize

# --- loading the input data ---
# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]
df.reset_index(drop=True, inplace=True)

# --- normalization of queries ---
# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
print("Normalize queries:)
normalized_df = df
if do_normalize:
    normalized_ls_of_dict = []
    for i in range(0, df.shape[0]):
        if i % 100000 == 0:
            print("Working on query " + str(i) + " ...")
        query = df["query"][i]
        normalized_query = df["query"][i].lower()
        normalized_query = re.sub("[^a-z0-9]" , " ", normalized_query)
        normalized_query = re.sub("\s+" , " ", normalized_query)
    #   if len(normalized_query) != len(query):
    #       print("original query: '" + query + "'")
    #       print("normalized query: '" + normalized_query + "'")
        dictionary_to_add = {"category": df["category"][i], "query": df["query"][i].lower()}
        normalized_ls_of_dict.append(dictionary_to_add)
    #    pp.pprint(dictionary_to_add)
    print("Normalization completed.")
    normalized_df = pd.DataFrame(normalized_ls_of_dict)
    #pp.pprint(normalized_df)
    #exit(0)
print("Completed normalizing queries.")

# --- rolling up of categories ---
# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
print("Rolling up categories...")

# - parent_dict: store category parent relation in dictionary -
parents_dict = {}
for i in range(0, parents_df.shape[0]):
    parents_dict[parents_df.category[i]] = parents_df.parent[i]

# - category_count_dict: calculate query count per category and store it in dictionary -
category_count_df = normalized_df.groupby("category", as_index=False)["category"].agg({"count":"count"})
category_count_dict = {}
for i in range(0, category_count_df.shape[0]):
    category_count_dict[category_count_df["category"][i]] = category_count_df["count"][i]
#pp.pprint(category_count_dict)
#exit(0)

# - roll up -
category_replacements_dict = {}
for category in parents_dict.keys():
    category_replacements_dict[category] = category # original status: each category stays
current_category_count_dict = copy.deepcopy(category_count_dict)
current_parents_dict = copy.deepcopy(parents_dict)

#category_to_check = "pcmcat237000050015"
#check_category(category_to_check, current_parents_dict, current_category_count_dict)

while True: # outer loop: continue as long as there are leaf categories removed / rolled up
    did_deletion = False
    for category in current_parents_dict.keys(): # inner loop: loop over leaf categories
        if category not in current_parents_dict.values():
            # only consider leaf categories
            current_parent = current_parents_dict[category]
            current_final_parent = current_parent
            if (current_parent in category_replacements_dict.keys()) and (current_parent != category_replacements_dict[current_parent]):
                current_final_parent = category_replacements_dict[current_parent]
            current_count = 0
            if category in current_category_count_dict.keys():
                current_count = current_category_count_dict[category]
            current_final_parent_count = 0
            if current_final_parent in current_category_count_dict.keys():
                current_final_parent_count = current_category_count_dict[current_final_parent]
            if ((current_count < min_queries) or
               ((current_final_parent_count > 0) and (current_final_parent_count < min_queries))):
                # remove current leaf category if it has none or too few queries
                # or if it has a final parent which has queries but too few
                del current_parents_dict[category] # remove leaf from structure
                did_deletion = True
                if category in current_category_count_dict.keys():
                    # only do full roll up if a count entry exists for the current category
 #                   print("rolling up '{}' via '{}' to '{}'. It has {} queries and a final parent with {} queries...".format(
 #                       category, current_parent, current_final_parent, current_count, current_final_parent_count
 #                   ))
                    current_category_count_dict[current_final_parent] =  current_final_parent_count + current_count
 #                   print("The parent category '{}' now has {} := {}+{} entries".format(
 #                       current_final_parent,  current_final_parent_count + current_count, current_final_parent_count, current_count
 #                   ))
                    affected_categories = [c for c, r in category_replacements_dict.items() if r == category]
                    for affected_category in affected_categories:
                        category_replacements_dict[affected_category] = current_parent
                    del current_category_count_dict[category]
                break # stop for loop if we have done a deletion
    if not did_deletion:
        break

#check_category(category_to_check, current_parents_dict, current_category_count_dict)
#pp.pprint(current_category_count_dict)
print("roll up is completed. {} categories were rolled up into {} categories.".format(
    len(category_count_dict.keys()), len(current_category_count_dict.keys()))
)

# apply roll up to dataframe of normalized queries and their categories
normalized_rolled_up_ls_of_dict = []
for i in range(0, normalized_df.shape[0]):
    category = normalized_df["category"][i]
    query = normalized_df["query"][i]
    rolled_up_category = category
    if category in category_replacements_dict.keys():
        rolled_up_category = category_replacements_dict[category]
    normalized_rolled_up_ls_of_dict.append(
        {"category": rolled_up_category, "query": query}
    )
df = pd.DataFrame(normalized_rolled_up_ls_of_dict)

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
print("normalized queries with rolled-up categories were written to '{}'".format(output_file_name))