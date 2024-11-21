```markdown
# Car Price Prediction with Machine Learning

This project focuses on predicting car prices using various machine learning techniques. The price of a car depends on multiple factors such as the brand, features, horsepower, mileage, and more. By analyzing these factors, we aim to build a model capable of accurately predicting car prices.

---

## Dataset

- **Source**: [Car Price Prediction Dataset](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)
- **Description**: Contains details about cars such as year, selling price, present price, kilometers driven, fuel type, selling type, transmission type, and ownership history.

---

## Project Workflow

### 1. **Data Preprocessing**
- Load the dataset using Pandas.
- Check for missing values and data types.
- Perform exploratory data analysis (EDA) to understand data distribution and relationships.

### 2. **Exploratory Data Analysis (EDA)**
- **Visualizations**:
  - Distribution of Selling Price.
  - Relationship between Car Age and Selling Price.
  - Present Price vs Selling Price.
  - Driven Kilometers vs Selling Price.

### 3. **Feature Engineering**
- Calculate car age from the manufacturing year.
- Separate the dataset into features (`X`) and target (`y`).
- Identify categorical and numerical features.

### 4. **Model Training**
- Use a `ColumnTransformer` to preprocess numerical and categorical features:
  - Scale numerical data with `StandardScaler`.
  - Encode categorical data with `OneHotEncoder`.
- Train the following models:
  - Linear Regression
  - Random Forest Regressor

### 5. **Model Evaluation**
- Evaluate the models using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Results for **Random Forest**:
  - **MAE**: 0.595
  - **MSE**: 0.789
  - **RMSE**: 0.888
  - **R² Score**: 0.966

---

## Technologies Used
- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`
- **Machine Learning Models**: 
  - Linear Regression
  - Random Forest Regressor

---

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone <repository-link>
   cd car-price-prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   - Place the dataset (`car data.csv`) in the project directory.

4. **Run the Notebook**:
   - Open `Car Price Prediction with Machine Learning.ipynb` in Jupyter Notebook or Google Colab.
   - Execute the cells sequentially.

---

## Results
The **Random Forest Regressor** performed the best with an R² score of **0.966**, indicating a strong ability to predict car prices accurately.

---

## Model Deployment
The trained model is saved as `car_price_model.pkl` for future use. You can use this file to make predictions without retraining the model.

---

## Author
- **Name**: D. Muni Tejo Venkata Sai  
- **Email**: [doosettytejesh@gmail.com](mailto:doosettytejesh@gmail.com)  

---
``` 

This README provides a detailed overview of your project, from objectives to implementation and results. Let me know if you'd like to add or modify anything!
