import argparse


def estimate_inexact_fscore(y_true, y_pred, b_equals_i=False):
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in range(len(y_true)):
        if b_equals_i:
            if y_true[i] == "B" or y_true[i] == "I":
                if y_pred[i] == "B" or y_pred[i] == "I":
                    tp += 1
                else:
                    fn += 1
            if y_true[i] == "O":
                if y_pred[i] == "O":
                    tn += 1
                else:
                    fp += 1
        else:
            if y_true[i] == "B":
                if y_pred[i] == "B":
                    tp += 1
                if y_pred[i] == "I":
                    fp += 1
                if y_pred[i] == "O":
                    fn += 1
            if y_true[i] == "I":
                if y_pred[i] == "B":
                    fp += 1
                if y_pred[i] == "I":
                    tp += 1
                if y_pred[i] == "O":
                    fn += 1
            if y_true[i] == "O":
                if y_pred[i] == "B":
                    fp += 1
                if y_pred[i] == "I":
                    fp += 1
                if y_pred[i] == "O":
                    tn += 1
    return tp, tn, fp, fn

def estimate_exact_fscore(y_true, y_pred):
    pairs = []
    start = False
    end = False
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in range(len(y_true)):
        if y_true[i] == "O":
            if y_pred[i] == "O":
                tn += 1
            else:
                fp += 1
        if (y_true[i] == "B"):
            if start:
                end_index = i
                start = False
                pairs.append((start_index, end_index)) 
            if i < len(y_true) - 1:
                if y_true[i+1] == "B":
                    start_index = i
                    pairs.append((start_index,))
                    continue
            else:
                if y_true[i] == "B":
                    start_index = i
                    pairs.append((start_index,))
                    continue
        if (y_true[i] == "B") and (not start):
            start = True
            start_index = i
        elif start and ((y_true[i] == "O") or (i == len(y_true) - 1)):
            start = False
            if i == len(y_true) - 1:
                end_index = i + 1
                pairs.append((start_index, end_index))
            else:
                end_index = i
                pairs.append((start_index, end_index))
    for pair in pairs:
        if len(pair) == 1:
            if y_true[pair[0]] == y_pred[pair[0]]:
                tp += 1
            else:
                fn += 1
        if len(pair) == 2:
            if y_true[pair[0]:pair[1]] == y_pred[pair[0]:pair[1]]:
                tp += 1
            else:
                fn += 1
    
    return tp, tn, fp, fn

def estimate_precision(tp, fp):
    return float(tp) / (tp + fp)

def estimate_recall(tp, fn):
    return float(tp) / (tp + fn)

def estimate_fscore(precision, recall):
    return float(2 * precision * recall) / (precision + recall)

def build_data_sequence(file_name):

    f = open(file_name, "r")

    sentence = []
    y_true = []
    y_pred = []

    data_sequence = []

    for line in f:
        tokens = line.split()
        if len(tokens) != 0:
            sentence.append(tokens[0])
            y_true.append(tokens[1])
            y_pred.append(tokens[2])
        else:
            data_sequence.append([sentence, y_true, y_pred])
            sentence = []
            y_true = []
            y_pred = []

    if sentence:
        data_sequence.append([sentence, y_true, y_pred])
        sentence = []
        y_true = []
        y_pred = []

    return data_sequence

def report_fscore(predictions_file):

    data_sequence = build_data_sequence(predictions_file)
        
    tp_exact = 0
    tn_exact = 0
    fp_exact = 0
    fn_exact = 0

    tp_inexact = 0
    tn_inexact = 0
    fp_inexact = 0
    fn_inexact = 0

    for _, y_true, y_pred in data_sequence:
        tp, tn, fp, fn = estimate_inexact_fscore(y_true, y_pred)
        tp_inexact += tp
        tn_inexact += tn
        fp_inexact += fp
        fn_inexact += fn
        tp, tn, fp, fn = estimate_exact_fscore(y_true, y_pred)
        tp_exact += tp
        tn_exact += tn
        fp_exact += fp
        fn_exact += fn

    print("--------------------- Prediction Statistics ---------------------\n")
    print("TP exact: ", tp_exact)
    print("TN exact: ", tn_exact)
    print("FP exact: ", fp_exact)
    print("FN exact: ", fn_exact)
    print()
    print("TP inexact: ", tp_inexact)
    print("TN inexact: ", tn_inexact)
    print("FP inexact: ", fp_inexact)
    print("FN inexact: ", fn_inexact)
    print()

    print("--------------------- Prediction Performance ---------------------\n")
    precision = estimate_precision(tp_exact, fp_exact)
    recall = estimate_recall(tp_exact, fn_exact)
    fscore = estimate_fscore(precision, recall)
    print("Exact matching:")
    print("f-score: {0:.3f}".format(fscore))
    print("precision: {0:.3f}".format(precision))
    print("recall: {0:.3f}".format(recall))
    print()

    precision = estimate_precision(tp_inexact, fp_inexact)
    recall = estimate_recall(tp_inexact, fn_inexact)
    fscore = estimate_fscore(precision, recall)
    print("Inexact matching:")
    print("f-score: {0:.3f}".format(fscore))
    print("precision: {0:.3f}".format(precision))
    print("recall: {0:.3f}".format(recall))
    print()

# if __name__ == "__main__":
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument("--file_name", type=str, help="path to predictions", required=True)
#     parsed_args = argparser.parse_args()
#     main(parsed_args)
