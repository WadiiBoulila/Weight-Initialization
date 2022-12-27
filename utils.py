
def performance_report(cm, mode='macro', printing=False):
    col = len(cm)
    labels = list(cm.keys())
    # col=number of class
    arr = []
    for key, value in cm.items():
        arr.append(value)
    cr = dict()
    support_sum = 0
    # macro avg of support is
    # sum of support only, not the mean.
    macro = [0] * 3  
    # weighted avg of support is
    # sum of support only, not the mean.
    weighted = [0] * 3
    for i in range(col):
        vertical_sum = sum([arr[j][i] for j in range(col)])
        horizontal_sum = sum(arr[i])
        p = arr[i][i] / vertical_sum
        r = arr[i][i] / horizontal_sum
        f = (2 * p * r) / (p + r)
        s = horizontal_sum
        row = [p,r,f,s]
        support_sum += s
        for j in range(3):
            macro[j] += row[j]
            weighted[j] += row[j]*s
        cr[i] = row
    # add Accuracy parameters.
    truepos=0
    total=0
    for i in range(col):
        truepos += arr[i][i]
        total += sum(arr[i])
    cr['Accuracy'] = ["", "", truepos/total, support_sum]
    # Add macro-weight features.
    macro_avg = [Sum/col for Sum in macro]
    macro_avg.append(support_sum)
    cr['Macro_avg'] = macro_avg
    # Add weighted_avg
    weighted_avg = [Sum/support_sum for Sum in weighted]
    weighted_avg.append(support_sum)
    cr['Weighted_avg'] = weighted_avg
    # print the classification_report
    if printing:
        stop=0
        max_key = max(len(str(x)) for x in list(cr.keys())) + 15
        print("Performance report of the model is :")
        print(f"%{max_key}s %9s %9s %9s %9s\n" % (" ", "Precision", "Recall", "F1-Score", "Support"))
        for i, (key, value) in enumerate(cr.items()):
            if stop<col:
                stop += 1
                print(f"%{max_key}s %9.2f %9.2f %9.2f %9d" % (labels[key] if isinstance(key, int) else key, value[0], value[1], value[2], value[3]))
            elif stop == col:
                stop += 1
                print(f"\n%{max_key}s %9s %9s %9.2f %9d" % (labels[key] if isinstance(key, int) else key, value[0], value[1], value[2], value[3]))
            else:
                print(f"%{max_key}s %9.2f %9.2f %9.2f %9d" % (labels[key] if isinstance(key, int) else key, value[0], value[1], value[2], value[3]))
    if mode == 'macro':
        return cr['Macro_avg']
    else:
        return cr['Weighted_avg']


def cm_to_dict(cm, labels):
    cm_dict = dict()
    for i, row in enumerate(cm):
        cm_dict[labels[i]] = row
    return cm_dict
