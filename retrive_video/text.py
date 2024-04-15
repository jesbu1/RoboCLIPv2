import pickle
file_name = "droid_100_text.pkl"
with open(file_name, 'rb') as file:
    text_dict = pickle.load(file)
    import pdb ; pdb.set_trace()