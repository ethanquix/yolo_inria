from network.darknet import Darknet
from data.mongoann import MongoAnn
from torch.utils.data.dataloader import DataLoader
import argparse
import ast
from timeEstimator import TimeEstimator
'''
TEST = '{"$match" : {"caltechsubset" : "1x-test", "anntype" : "new","occlusion" : {"$lte" : 0.35}, "height" : {"$gte": 50}, "object" :"person"} },{"$group" : { "_id": "$image","information" : { "$push" :{"object" :' "$object", "gt" : "$gtfull"}}}}'
TRAINING = '{"$match" : {"caltechsubset" : "10x-train", "anntype" : "new","occlusion" : {"$lte" : 0.35}, "height" : {"$gte": 50}, "object" :"person"} },{"$group" : { "_id": "$image","information" : { "$push" :{"object" :' "$object", "gt" : "$gtfull"}}}}'
'''
parser = argparse.ArgumentParser(
    description='YOLO V1 Training')

parser.add_argument('--query')
parser.add_argument('--querytest')
parser.add_argument('--classmap')
parser.add_argument('--bs')
parser.add_argument('--workers')


def main():
    args = parser.parse_args()
    query = args.query
    querytest = args.querytest
    classmap = ast.literal_eval(args.classmap)

    model = Darknet('/home/dwyzlic/projects/myyolo/cfg/yolo.cfg')

    model.load_weights('/home/dwyzlic/projects/myyolo/yolov3.weights')

    trainDB = MongoAnn(query, classmap)
    testDB = MongoAnn(querytest, classmap)

    print('db done')

    trainDB = DataLoader(trainDB, shuffle=True, batch_size=int(args.bs), num_workers=int(args.workers))
    testDB = DataLoader(testDB, shuffle=True, batch_size=int(args.bs), num_workers=int(args.workers))

    model.train()

    maxEpoch = 10
    epoch = 0

    TEepoch = TimeEstimator(maxEpoch)
    TEbatch = TimeEstimator(len(trainDB))

    while epoch < maxEpoch:
        TEepoch.start()
        print('Epoch ' + str(epoch))

        # TRAIN
        for num_batch, batch in enumerate(trainDB):
            image, bboxes, labels = batch

            print('[TRAIN] num batch ' + str(num_batch))
            prediction = model(image, True)
            print('[TRAIN] done num batch ' + str(num_batch))

        # TEST
        for num_batch, batch in enumerate(testDB):
            image, bboxes, labels = batch

            print('[TEST] num batch ' + str(num_batch))
            prediction = model(image, True)
            print('[TRAIN] done num batch ' + str(num_batch))
            print(prediction)

        print('Time remaining for epoch: ' + str(TEepoch))


if __name__ == '__main__':
    main()
