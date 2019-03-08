import pickle;



def save(variable, flieName):
    with open(flieName, 'wb') as f:
        pickle.dump(variable, f);


def load(flieName):
    with open(flieName, 'rb') as f:
        b = pickle.load(f)
        return b;
