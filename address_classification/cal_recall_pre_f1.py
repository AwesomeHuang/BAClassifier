class _MD(object):
    mapper = {
        str: '',
        int: 0,
        list: list,
        dict: dict,
        set: set,
        bool: False,
        float: .0
    }

    def __init__(self, obj, default=None):
        self.dict = {}
        assert obj in self.mapper, \
            'got a error type'
        self.t = obj
        if default is None:
            return
        assert isinstance(default, obj), \
            f'default ({default}) must be {obj}'
        self.v = default

    def __setitem__(self, key, value):
        self.dict[key] = value


    def __getitem__(self, item):
        if item not in self.dict and hasattr(self, 'v'):
            self.dict[item] = self.v
            return self.v
        elif item not in self.dict:
            if callable(self.mapper[self.t]):
                self.dict[item] = self.mapper[self.t]()
            else:
                self.dict[item] = self.mapper[self.t]
            return self.dict[item]
        return self.dict[item]

def defaultdict(obj, default=None):
    return _MD(obj, default)


def cal_precision_and_recall(true_labels, pre_labels):
    precision = defaultdict(int, 1)
    recall = defaultdict(int, 1)
    total = defaultdict(int, 1)
    for t_lab, p_lab in zip(true_labels, pre_labels):
        total[t_lab] += 1
        recall[p_lab] += 1
        if t_lab == p_lab:
            precision[t_lab] += 1
    
    for sub in precision.dict:
        pre = precision[sub] / recall[sub]
        rec =  precision[sub] / total[sub]
        F1 = (2 * pre * rec) / (pre + rec)
        print(f"{str(sub)}  precision: {str(pre)}  recall: {str(rec)}  F1: {str(F1)}")