USE customer_churn_db;

CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    gender VARCHAR(20),
    tenure INT,
    contact VARCHAR(50),
    monthly_charges FLOAT,
    total_charges FLOAT,
    prediction INT,
    churn_probability FLOAT,
    created_at TIMESTAMP DEFAULT
    CURRENT_TIMESTAMP
);