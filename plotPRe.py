precision=[0.8512, 0.8575, 0.9418, 0.8594, 0.8524, 0.8621, 0.8960, 0.8582, 0.8798,
        0.7031, 0.7121, 0.7988, 0.8482, 0.8784, 0.8064, 0.8477, 0.8107]
recall=[0.8361, 0.8463, 0.9434, 0.8521, 0.8535, 0.8668, 0.8953, 0.8454, 0.9685,
        0.7208, 0.6742, 0.8457, 0.8450, 0.8657, 0.7955, 0.8660, 0.7488]
def calculate_f1_scores(precision_list, recall_list):
    f1_scores = []
    for precision, recall in zip(precision_list, recall_list):
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)
    return f1_scores

# Example usage:

f1_scores = calculate_f1_scores(precision, recall)
print("F1 Scores:", f1_scores)