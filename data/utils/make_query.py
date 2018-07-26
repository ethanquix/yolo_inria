"""
Example usage :

python make_query.py --query '''{"$match" : {"caltechsubset" :
 "10x-train", "anntype" : "new","occlusion" : {"$lte" : 1.0}, "height" :
  {"$gte": -1}, "object" : "person"} },{"$group" : { "_id": "$image",
  "information" : { "$push" : {"object" : "$object", "gt" : "$gtfull"}}}}'''
"""

import argparse
import ast

import cv2
from pymongo import MongoClient

parser = argparse.ArgumentParser()
parser.add_argument('--query')
parser.add_argument('--classmap')


def make_query(query):
    results = []
    client = MongoClient('nef-devel2.inria.fr', 27017)
    db = client.pedestrians
    collections = db.collection_names(include_system_collections=False)
    for collection in collections:
        if collection == 'all':
            continue
        coll = db[collection]
        arg_agg = ast.literal_eval('''{}'''.format(query))
        resultcur = coll.aggregate([arg_agg[0], arg_agg[1]])
        resultcur = list(resultcur)
        results += resultcur
    return results


def groupbyimages(results, classdict, format='JPEG'):
    for ind in range(len(results)):
        results[ind]['filename'] = results[ind].pop('_id')
        '''
        img = cv2.imread(results[ind]['filename'])
        height, width, _ = img.shape
        results[ind]['height'] = height
        results[ind]['width'] = width
        '''
        results[ind]['format'] = format
        results[ind]['id'] = ind
        results[ind]['object'] = {'bbox': {'text': [], 'label': [],
                                           'xmin': [], 'xmax': [],
                                           'ymin': [], 'ymax': []}}
        for item in results[ind]['information']:
            results[ind]['object']['bbox']['text'].append(item['object'])
            results[ind]['object']['bbox']['label'].append(classdict[item[
                'object']])
            results[ind]['object']['bbox']['xmin'].append(item['gt'][0])
            results[ind]['object']['bbox']['xmax'].append(item['gt'][2])
            results[ind]['object']['bbox']['ymin'].append(item['gt'][1]
                                                          )
            results[ind]['object']['bbox']['ymax'].append(item['gt'][3]
                                                          )
        results[ind].pop('information')

    return results


if __name__ == "__main__":
    args = parser.parse_args()
    query = args.query
    classmap = args.classmap
    classmap = ast.literal_eval(classmap)
    results = make_query(query)
    results = groupbyimages(results, classmap)
