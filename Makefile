all:
		python3 -u run.py --query '{"$match" : {"caltechsubset" : "10x-train", "anntype" : "new","occlusion" : {"$lte" : 0.35}, "height" : {"$gte": 50}, "object" :"person"} },{"$group" : { "_id": "$image","information" : { "$push" :{"object" : "$object", "gt" : "$gtfull"}}}}' --classmap '{"person": 1}' --bs 1 --workers 5 --querytest '{"$match" : {"caltechsubset" : "10x-train", "anntype" : "new","occlusion" : {"$lte" : 0.35}, "height" : {"$gte": 50}, "object" :"person"} },{"$group" : { "_id": "$image","information" : { "$push" :{"object" : "$object", "gt" : "$gtfull"}}}}'

setup:
		$(info Type:)
		$(info module load cuda/9.1 cudnn/7.0-cuda-9.1 /home/uujjwal/ujjwal-modules/openmpi-3.0.0 myopencv mypytorch gcc/5.3.0 && conda activate pytorch)
