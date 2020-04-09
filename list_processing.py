import re, random
from sklearn.model_selection import KFold

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
    any args can be applied
    -------
    example
    -------
    a = [[1],[2],[3],[4]]
    b = [[5],[6],[7],[8]]
    c = [[9],[10],[11],[12]]
    
    a,b,c = list_shuffle(a,b,c)
    print(a)
    print(b)
    print(c)
    
    -------
    result
    -------    
    ([2], [4], [3], [1])
    ([6], [8], [7], [5])
    ([10], [12], [11], [9])
    
    """
    seed=1
    combined = list(zip(*args))
    random.seed(seed)
    random.shuffle(combined)

    return zip(*combined)

def list_split_KFold(x_list,y_list,Fold_total,Fold_selected):
    """
    example
    Fold_total = 5
    Fold_selected = 1
    return x_list, y_list
    """
    
    assert Fold_total >= Fold_selected
    assert Fold_selected >=1
    
    print('Fold:',Fold_selected,'/',Fold_total)
    kf = KFold(n_splits=Fold_total,shuffle=True,random_state=1)
    kf.get_n_splits(x_list)

    count = 0
    
    for train_index, test_index in kf.split(x_list):
        
        x_train, y_train = [], []
        x_valid, y_valid = [], []

        for idx in range(len(train_index)):
            x_train.append(x_list[train_index[idx]])
            y_train.append(y_list[train_index[idx]])
            
        for idx in range(len(test_index)):
            x_valid.append(x_list[test_index[idx]])
            y_valid.append(y_list[test_index[idx]])
        
        count += 1
        if count == Fold_selected:        
            print("TRAIN:", train_index, "TEST:", test_index)
            return x_train, y_train, x_valid, y_valid
