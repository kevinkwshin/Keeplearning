import re, random

def list_sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]

    l.sort(key=alphanum_key)
    return l

def list_shuffle(*args):
    """
    LISTS = ([0,1],[1,0]) or ((0,1),(1,0))
    """
    seed=1
    combined = list(zip(*args))
    random.seed(seed)
    random.shuffle(combined)

    return zip(*combined)

def list_split_train(list1,list2,rate=0.4):
    
    x_train_  = list1[:-int(len(list1)*rate)]
    y_train_  = list2[:-int(len(list2)*rate)]
    x_tuning_ = list1[-int(len(list1)*rate):]
    y_tuning_ = list2[-int(len(list2)*rate):]
    
    return x_train_,y_train_,x_tuning_,y_tuning_
