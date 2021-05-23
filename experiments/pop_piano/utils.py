import pickle

def pickle_load(f):
  return pickle.load(open(f, 'rb'))

def pickle_dump(obj, f):
  pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)