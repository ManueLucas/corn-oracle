import pickle
import pprint

filename = '.\models\eval_res.pkl'
with open(filename, 'rb') as file:
    eval_res = pickle.load(file)

    pprint.pprint(eval_res)