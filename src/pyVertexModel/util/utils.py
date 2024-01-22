import pickle


def save_state(obj, filename):
    """
    Save state of the different attributes of obj in filename
    :param obj:
    :param filename:
    :return:
    """
    with open(filename, 'wb') as f:
        # Go through all the attributes of obj
        for attr in dir(obj):
            # If the attribute is not a method, save it
            if not callable(getattr(obj, attr)) and not attr.startswith("__"):
                print(attr)
                pickle.dump({attr: getattr(obj, attr)}, f)


def load_state(obj, filename):
    """
    Load state of the different attributes of obj from filename
    :param obj:
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                for attr, value in data.items():
                    setattr(obj, attr, value)
            except EOFError:
                break
