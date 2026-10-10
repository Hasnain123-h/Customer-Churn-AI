from database.connection import get_connection

def save_prediction(
    gender,
    tenure,
    contract,
    monthly_charges,
    total_charges,
    prediction,
    churn_probability
):

    connection = get_connection()
    cursor = connection.cursor()
    query = """
    INSERT INTO predictions
    (
        gender,
        tenure,
        contract,
        monthly_charges,
        total_charges,
        prediction,
        churn_probability
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        gender,
        tenure,
        contract,
        monthly_charges,
        total_charges,
        prediction,
        churn_probability
    )

    cursor.execute(query, values)
    connection.commit()
    cursor.close()
    connection.close()
def get_predictions():
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM predictions ORDER BY id DESC"

    cursor.execute(query)
    predictions = cursor.fetchall()

    cursor.close()
    connection.close()
    return predictions


if __name__ == "__main__":
    print("Database queries module is working.")