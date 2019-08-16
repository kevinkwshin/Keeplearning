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

def list_shuffle(list1,list2):
    
    combined = list(zip(list1,list2))
    random.shuffle(combined)
    list1,list2 = zip(*combined)
    list1 = list(list1)
    list2 = list(list2)
    
    return list1, list2
