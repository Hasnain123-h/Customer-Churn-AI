def get_churn_label(prediction):
    if prediction == 1:
        return "Churn"
    else:
        return "No Churn"


def get_churn_probability(probability):
    return f"{probability * 100:.2f}%"