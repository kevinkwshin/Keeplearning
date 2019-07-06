def shuffle_list(list1,list2):
    combined = list(zip(list1,list2))
    random.shuffle(combined)
    list1,list2 = zip(*combined)
    list1 = list(list1)
    list2 = list(list2)
    return list1, list2
