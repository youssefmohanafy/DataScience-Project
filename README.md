# Data Science Project Workflow

## Project Overview
- **Objective**: Define the problem and goals of the project.
- **Milestones**: Data Collection, Exploration, Preprocessing, Advanced Analysis, Model Development, Deployment, and Final Documentation.

---
## Domain and Research Questions

### Domain of the Project
- Online retail & Customer behaviour.
### Research Questions to be Answered
1.  Which product categories generate the highest revenue, and what factors contribute to their success?  
2.  What is the impact of different marketing strategies (discounts, advertisements, email campaigns) on customer conversion rates?  
3.	How do seasonal trends, including major holidays and sales events, impact e-commerce revenue?

---
# Team Information

## Student Information
- **Name**: Youssef Mohamed Hanafy Mahmoud  
- **Email**: yh2000009@tkh.edu.eg 
- **Role**: Data Science Student  


## Additional Information
- **Project Timeline**: [Insert Start Date - End Date]  
- **Tools Used**: [Insert List of Tools or Frameworks, e.g., Python, SQLite, Pandas, etc.]  
- **Advisor/Instructor**: [Insert Advisor/Instructor Name, if applicable]  
- **Contact for Inquiries**: [Insert Email or Point of Contact]

---
# Milestone 1: Data Collection, Exploration, and Preprocessing
## Data Collection
- Acquire a dataset from reliable sources (e.g., Kaggle, UCI Repository, or APIs).
- **Scraping Data**:
  - Increase dataset size through web scraping or APIs (e.g., Selenium, BeautifulSoup).
  - Explore public repositories or other accessible sources for additional data.
## Import Necessary Libraries & Load Datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the datasets (update file paths if needed)
ecommerce_data = pd.read_csv('E-commerce Dataset.csv')  # E-commerce transactions
noon = pd.read_csv('noon_dataset.csv')  # Noon dataset
noon_cleaned = pd.read_csv('noon_dataset_cleaned.csv')  # Cleaned Noon dataset

# Display basic dataset information
print("E-Commerce Dataset:")
print(ecommerce_data.info())

print("\n noon_dataset:")
print(noon.info())
## Preview the Data
print("\nE-Commerce Dataset Sample:")
display(ecommerce_data.head())



print("\nnoon_dataset  Sample:")
display(noon.head())

## Dataset Description

### **1. E-Commerce Dataset.csv**

51,290 entries, 16 columns.
Contains purchase transactions with details such as:
- **Order_Date, Time**: Transaction timestamps.
- **Aging**: The time from the day the product is ordered to the day it is delivered
- **Customer_Id**: Unique id created for each customer.
- **Gender, Device_Type, Customer_Login_type**: Customer attributes.
- **Product_Category, Product**: Items purchased.
- **Sales, Quantity, Discount, Profit, Shipping_Cost**: Financial details.
- **Order_Priority**: Order priority. Such as critical, high etc.
- **Payment_method**: Payment method.

This dataset helps analyze:
- **User behavior and engagement trends**.
- **Purchase patterns across devices and countries**.
- **Product popularity and sales performance**.

### **2. noon_dataset.csv** Scraped
30,375 entries, 2 colunms.
- **Product name**
- **Price**

## Data Exploration
- Summary statistics (mean, median, variance).
- Identify missing values, duplicates, and outliers.
- Data distribution visualizations: histograms, box plots, scatter plots.
## E-commerce visualisation before cleaning
import pandas as pd

# Load the raw dataset
ecommerce_dataset = pd.read_csv('E-commerce Dataset.csv')
noon_dataset = pd.read_csv('noon_dataset.csv')


# Display data types of each column
ecommerce_data.info()
noon_dataset.info()

# Show the first few rows of the dataset to understand the formats
ecommerce_dataset.head()
noon_dataset.head()


#### Date and Time Conversion 
# Convert 'Order Date' to datetime
ecommerce_dataset['Order_Date'] = pd.to_datetime(ecommerce_dataset['Order_Date'])

# Convert 'Time' to a proper time format if needed
ecommerce_dataset['Time'] = pd.to_timedelta(ecommerce_dataset['Time'])

# Check the new data types to confirm the conversion
print("Updated Data Types:")
print(ecommerce_dataset.dtypes)

# Show the first few rows to verify changes
print("\nUpdated Data Preview:")
print(ecommerce_dataset.head())

#### Identify numerical and categorical columns in E-commerce Dataset
# Identify numerical and categorical columns
numerical = [var for var in ecommerce_dataset.columns if ecommerce_dataset[var].dtype != "O"]
categorical = [var for var in ecommerce_dataset.columns if ecommerce_dataset[var].dtype == "O"]

print(f"Numerical Variables: {numerical}")
print(f"Categorical Variables: {categorical}")
#### Histograms
import plotly.express as px
# Generate histograms for numerical columns using Plotly
for var in numerical:
    hist = px.histogram(ecommerce_dataset, x=var, nbins=50, title=f"{var} Distribution",
                        marginal='violin', # This adds a violin plot to the histogram for a better understanding of the distribution
                        template='plotly_dark'
                        )
    hist.show()
#### My observations on the histograms created:
- Order_Date:
Sales increased over time, with some seasonal peaks.

- Time:
Most orders happen later in the day, with fewer in early hours.

- Aging:
Orders take a consistent number of days for delivery.

- Customer_Id:
Customers are evenly distributed over time.

- Sales:
Sales have two main peaks, likely due to different product price ranges.

- Quantity:
Most customers buy only one item per order.

- Discount:
Discounts are given at fixed rates like 10%, 20%, etc.

- Profit:
Most sales have low profit, but some generate high profit.

- Shipping_Cost:
Most orders have low shipping costs, with another peak around 10-15.


#### Barplot
import matplotlib.pyplot as plt
import seaborn as sns

# Generate bar plots for categorical columns
for col in categorical:
    plt.figure(figsize=(12, 6))

    # Limit to top 10 categories if too many unique values
    value_counts = ecommerce_dataset[col].value_counts().nlargest(9)

    sns.barplot(x=value_counts.index, y=value_counts.values, palette="Blues_r")
    plt.title(f"{col} Distribution")
    plt.ylabel("Count")
    
    plt.show()

#### My opservations on the barplot created:
- Order Date: Most orders happened on 2018-04-24, then dropped steadily.
- Time: Orders mostly placed in the evening.
- Gender: More male customers than female.
- Device Type: Most people shop using the web, not mobile.
- Login Type: Most users are members; guests are very few.
- Product Category: Fashion is the most popular.
- Product: Suits, T-shirts, and watches are top-selling items.
- Order Priority: Medium priority is the most common.
- Payment Method: Credit card is used the most; others are less common.

### Scatter plot 
####  Logical and Beneficial Scatter Plot Ideas:
Sales vs Profit
- Business Insight: Are high sales always profitable?

Discount vs Sales
- Marketing Insight: Is discounting effective?

Shipping Cost vs Profit
- Logistics Insight: Do higher shipping costs reduce profitability?

Aging vs Profit or Aging vs Sales
- Operational Insight: Does delay affect sales/profit?
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create a 2x2 grid for subplots
fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Sales vs Profit", 
    "Discount vs Sales", 
    "Shipping Cost vs Profit", 
    "Aging vs Profit"
))

# Add Scatter Plot 1: Shows relationship between Sales and Profit
fig.add_trace(go.Scatter(x=ecommerce_dataset['Sales'], y=ecommerce_dataset['Profit'],
                         mode='markers', name='Sales vs Profit'),
              row=1, col=1)

# Add Scatter Plot 2: Shows effect of Discount on Sales
fig.add_trace(go.Scatter(x=ecommerce_dataset['Discount'], y=ecommerce_dataset['Sales'],
                         mode='markers', name='Discount vs Sales'),
              row=1, col=2)

# Add Scatter Plot 3: Checks if Shipping Cost affects Profit
fig.add_trace(go.Scatter(x=ecommerce_dataset['Shipping_Cost'], y=ecommerce_dataset['Profit'],
                         mode='markers', name='Shipping vs Profit'),
              row=2, col=1)

# Add Scatter Plot 4: Explores link between delivery time (Aging) and Profit
fig.add_trace(go.Scatter(x=ecommerce_dataset['Aging'], y=ecommerce_dataset['Profit'],
                         mode='markers', name='Aging vs Profit'),
              row=2, col=2)

# Customize overall layout: size, title, and theme
fig.update_layout(title_text="Scatter Plot Matrix", template='plotly_dark')

fig.show()

#### Observation
Sales vs Profit 
- Profit increases steadily with higher sales.

Discount vs Sales 
- Sales values are similar across all discount levels.

Shipping Cost vs Profit 
-  Higher shipping cost leads to higher profit.

Aging vs Profit 
- Profit stays consistent regardless of delivery time.
----
## Check for missing values in each dataset
# Load the datasets
noon_dataset = pd.read_csv('noon_dataset.csv')
ecommerce_dataset = pd.read_csv('E-commerce Dataset.csv')

# Check for missing values
missing_values = {
    "E-commerce Dataset": ecommerce_dataset.isnull().sum(),
    "Noon Dataset": noon_dataset.isnull().sum()
}

# Check for duplicates
duplicates = {
    "E-commerce Dataset": ecommerce_dataset.duplicated().sum(),
    "Noon Dataset": noon_dataset.duplicated().sum()
}

missing_values, duplicates

1. Missing Values

E-Commerce Dataset:
Aging (1 missing)
Sales (1 missing)
Quantity (2 missing)
Discount (1 missing)
Shipping_Cost (1 missing)
Order_Priority (2 missing)

Noon Dataset:
No missing data.


2. Duplicate Entries:

E-Commerce Dataset:
No Duplicates.

Noon Dataset:
5070


#### Cleaning
# Remove duplicates from Noon dataset
noon_dataset = noon_dataset.drop_duplicates()

# Drop rows with any missing values in E-commerce Dataset
ecommerce_dataset_cleaned = ecommerce_dataset.dropna()

# Save the cleaned data
ecommerce_dataset_cleaned.to_csv('E-commerce_Dataset_cleaned.csv', index=False)
noon_dataset.to_csv('noon_dataset_cleaned.csv', index=False)

# New structures after cleaning
cleaned_structures = {"E-commerce Dataset Cleaned": ecommerce_dataset_cleaned.head(),
                        "Noon Dataset": noon_dataset.head()}


cleaned_structures

# Load the cleaned datasets
ecommerce_dataset_cleaned = pd.read_csv('E-commerce_Dataset_cleaned.csv')
noon_dataset = pd.read_csv('noon_dataset_cleaned.csv')

# Display the first few rows to confirm loading
loaded_cleaned_data = {
    "Noon Dataset Cleaned": noon_dataset.head(),
    "E-commerce Dataset Cleaned": ecommerce_dataset_cleaned.head()
}

loaded_cleaned_data

## E-commerce cleaned visualisation
import pandas as pd

# Load the raw dataset
ecommerce_dataset_cleaned = pd.read_csv('E-commerce Dataset.csv')

# Display data types of each column
ecommerce_dataset_cleaned.info()

# Show the first few rows of the dataset to understand the formats
ecommerce_dataset_cleaned.head()

#### Date and Time Conversion 
# Convert 'Order Date' to datetime
ecommerce_dataset_cleaned['Order_Date'] = pd.to_datetime(ecommerce_dataset_cleaned['Order_Date'])

# Convert 'Time' to a proper time format if needed
ecommerce_dataset_cleaned['Time'] = pd.to_timedelta(ecommerce_dataset_cleaned['Time'])

# Check the new data types to confirm the conversion
print("Updated Data Types:")
print(ecommerce_dataset_cleaned.dtypes)

# Show the first few rows to verify changes
print("\nUpdated Data Preview:")
print(ecommerce_dataset_cleaned.head())

#### Identify numerical and categorical columns
# Identify numerical and categorical columns
numerical = [var for var in ecommerce_dataset.columns if ecommerce_dataset[var].dtype != "O"]
categorical = [var for var in ecommerce_dataset.columns if ecommerce_dataset[var].dtype == "O"]

print(f"Numerical Variables: {numerical}")
print(f"Categorical Variables: {categorical}")
#### Histograms
import plotly.express as px
# Generate histograms for numerical columns using Plotly
for var in numerical:
    hist = px.histogram(ecommerce_dataset_cleaned, x=var, nbins=50, title=f"{var} Distribution",
                        marginal='violin', # This adds a violin plot to the histogram for a better understanding of the distribution
                        template='plotly_dark'
                        )
    hist.show()
#### Barplot
import matplotlib.pyplot as plt
import seaborn as sns

# Generate bar plots for categorical columns
for col in categorical:
    plt.figure(figsize=(12, 6))

    # Limit to top 10 categories if too many unique values
    value_counts = ecommerce_dataset[col].value_counts().nlargest(9)

    sns.barplot(x=value_counts.index, y=value_counts.values, palette="Blues_r")
    plt.title(f"{col} Distribution")
    plt.ylabel("Count")
    
    plt.show()

### Scatter plot 
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create a 2x2 grid for subplots
fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Sales vs Profit", 
    "Discount vs Sales", 
    "Shipping Cost vs Profit", 
    "Aging vs Profit"
))

# Add Scatter Plot 1: Shows relationship between Sales and Profit
fig.add_trace(go.Scatter(x=ecommerce_dataset_cleaned['Sales'], y=ecommerce_dataset_cleaned['Profit'],
                         mode='markers', name='Sales vs Profit'),
              row=1, col=1)

# Add Scatter Plot 2: Shows effect of Discount on Sales
fig.add_trace(go.Scatter(x=ecommerce_dataset_cleaned['Discount'], y=ecommerce_dataset_cleaned['Sales'],
                         mode='markers', name='Discount vs Sales'),
              row=1, col=2)

# Add Scatter Plot 3: Checks if Shipping Cost affects Profit
fig.add_trace(go.Scatter(x=ecommerce_dataset_cleaned['Shipping_Cost'], y=ecommerce_dataset_cleaned['Profit'],
                         mode='markers', name='Shipping vs Profit'),
              row=2, col=1)

# Add Scatter Plot 4: Explores link between delivery time (Aging) and Profit
fig.add_trace(go.Scatter(x=ecommerce_dataset_cleaned['Aging'], y=ecommerce_dataset_cleaned['Profit'],
                         mode='markers', name='Aging vs Profit'),
              row=2, col=2)

# Customize overall layout: size, title, and theme
fig.update_layout(title_text="Scatter Plot Matrix", template='plotly_dark')

fig.show()

----
## Preprocessing and Feature Engineering
- Handle missing values.
- Remove duplicates and outliers.
- Apply transformations (scaling, encoding, feature interactions).
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

# Load the cleaned dataset
ecommerce = pd.read_csv('E-commerce_Dataset_cleaned.csv')

# 1. Select the columns we want to process
# These are the numeric columns we will scale
numeric_features = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Aging']

# These are the text (categorical) columns we will convert to numbers
categorical_features = ['Gender', 'Device_Type', 'Customer_Login_type', 'Product_Category', 'Order_Priority', 'Payment_method']

# 2. Create a pipeline for numeric columns
# It will standardize (scale) the numbers so they have similar ranges
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Changes values to have mean 0 and std 1
])

# 3. Create a pipeline for categorical columns
# It will turn text values into 0s and 1s (one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Ignores unseen categories during testing
])

# 4. Combine both pipelines into one transformer
# This tells which pipeline to use for which columns
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),  # Use scaler for numeric columns
    ('cat', categorical_transformer, categorical_features)  # Use encoder for categorical columns
])

# 5. Apply the full preprocessing to the data
ecommerce_transformed = preprocessor.fit_transform(ecommerce)  # Fit and transform the dataset

# 6. Show the shape of the final result
print("Transformed dataset shape:", ecommerce_transformed.shape)  # Tells how many rows and new columns


---
# Milestone 2: Advanced Data Analysis and Feature Engineering
## Statistical Analysis
- Conduct tests such as t-tests, ANOVA, and chi-squared to explore relationships.
# Import libraries
import pandas as pd
from scipy.stats import f_oneway

# Load the cleaned dataset
ecommerce = pd.read_csv("E-commerce Dataset.csv")

# Group the 'Profit' column by 'Device_Type'
groups = [group["Profit"].values for name, group in ecommerce.groupby("Device_Type")]

# Apply ANOVA test
anova_result = f_oneway(*groups)

# Display the result
print("ANOVA F-statistic:", anova_result.statistic)
print("P-value:", anova_result.pvalue)

#### ANOVA tests between every numerical variable and every categorical variable
import pandas as pd
from scipy.stats import f_oneway

# Load dataset
ecommerce = pd.read_csv("E-commerce Dataset.csv")

# Define numerical and categorical columns
numerical_columns = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Aging']
categorical_columns = ['Gender', 'Device_Type', 'Customer_Login_type', 'Product_Category', 'Order_Priority', 'Payment_method']

# Loop through all combinations and apply ANOVA
for cat in categorical_columns:
    for num in numerical_columns:
        try:
            # Group numeric values by each category
            groups = [group[num].dropna().values for _, group in ecommerce.groupby(cat)]
            
            # Only test if we have 2 or more groups
            if len(groups) >= 2:
                stat, pval = f_oneway(*groups)
                print(f"ANOVA Test - {num} by {cat}: F-statistic = {stat:.3f}, p-value = {pval:.4f}")
        except Exception as e:
            print(f"Error testing {num} by {cat}: {e}")

#### Conclusion

- Product Category affects everything the most.
- Gender affects sales, quantity, discount, aging.
- Device Type affects quantity, discount, aging.
- Customer Login Type affects quantity and discount.

____

- Payment Method doesn’t affect anything.
- Customer Login Type doesn’t affect sales, profit, or aging.
- Device Type and Gender don’t impact profit or shipping cost much.

____

#### Strong Effects (p-value < 0.05)
- Product_Category affects everything:
Sales (F=1390.7, p=0.0000)

Profit (F=871.1, p=0.0000)

Discount (F=4744.4, p=0.0000)

Shipping_Cost (F=868.4, p=0.0000)

Quantity (F=99.3, p=0.0000)

Aging (F=114.5, p=0.0000)

- Gender:

Sales (F=9.018, p=0.0027)

Quantity (F=42.24, p=0.0000)

Discount (F=370.4, p=0.0000)

Aging (F=83.72, p=0.0000)

- Device_Type:

Quantity (F=47.95, p=0.0000)

Discount (F=11.93, p=0.0006)

Aging (F=10.13, p=0.0015)

- Customer_Login_type:

Quantity (F=4.845, p=0.0023)

Discount (F=13.41, p=0.0000)

#### No Significant Effect (p-value > 0.05)

- Payment_method:

Profit (F=1.12, p=0.3447)

Sales (not listed, but not significant)

Others (all p > 0.3)

- Gender:

Profit (F=3.10, p=0.0785)

Shipping_Cost (F=3.05, p=0.0809)

- Device_Type:

Sales (F=2.46, p=0.1169)

Profit (F=2.79, p=0.0951)

Shipping_Cost (F=2.75, p=0.0970)

- Customer_Login_type:

Profit (F=1.45, p=0.225)

Aging (F=1.35, p=0.257)
## Feature Engineering
- Create derived features based on domain knowledge.
- Apply transformations such as normalization, log scaling, or polynomial features.
import pandas as pd
import numpy as np

# Load the dataset
ecommerce = pd.read_csv("E-commerce Dataset.csv")

# 1. Create a new column to show how much was spent per item
ecommerce['Sales_per_Quantity'] = ecommerce['Sales'] / ecommerce['Quantity']

# 2. Create a new column to show profit margin (how much profit from each sale)
ecommerce['Profit_Margin'] = ecommerce['Profit'] / ecommerce['Sales']

# 3. Use log to make big numbers smaller and reduce skewness in data
ecommerce['Log_Sales'] = np.log1p(ecommerce['Sales'])  # log(1 + sales)
ecommerce['Log_Profit'] = np.log1p(ecommerce['Profit'])  # log(1 + profit)

# 4. Create squared values to catch non-linear patterns
ecommerce['Discount_Squared'] = ecommerce['Discount'] ** 2
ecommerce['Shipping_Cost_Squared'] = ecommerce['Shipping_Cost'] ** 2

# 5. Normalize 'Aging' so all values fall between 0 and 1 (helps some models)
ecommerce['Normalized_Aging'] = (ecommerce['Aging'] - ecommerce['Aging'].min()) / (ecommerce['Aging'].max() - ecommerce['Aging'].min())

# Show the first few rows of the new features
ecommerce[['Sales_per_Quantity', 'Profit_Margin', 'Log_Sales', 'Log_Profit', 
           'Discount_Squared', 'Shipping_Cost_Squared', 'Normalized_Aging']].head()

## Data Visualization
- Generate insightful visualizations:
  - Correlation heatmaps, pair plots.
  - Trends and comparisons using bar charts, line charts, and dashboards.
###  1. Which product categories generate the highest revenue?
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
ecommerce_dataset_cleaned = pd.read_csv("E-commerce Dataset.csv")

# Total Sales by Product Category
plt.figure(figsize=(12,6))
category_sales = ecommerce_dataset_cleaned.groupby('Product_Category')['Sales'].sum().sort_values(ascending=False)
sns.barplot(x=category_sales.index, y=category_sales.values, palette='Blues_r')
plt.title("Total Sales by Product Category")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()

# Profit distribution per category
plt.figure(figsize=(12,6))
sns.boxplot(data=ecommerce_dataset_cleaned, x='Product_Category', y='Profit')
plt.title("Profit Distribution by Product Category")
plt.xticks(rotation=45)
plt.show()

# Discount vs Sales by category (scatter)
plt.figure(figsize=(10,6))
sns.scatterplot(data=ecommerce_dataset_cleaned, x='Discount', y='Sales', hue='Product_Category')
plt.title("Discount vs Sales by Product Category")
plt.show()

### Conclusion
- Fashion has the highest sales and good, consistent profit.
- Home & Furniture and Auto & Accessories follow, but with less profit.
- Electronics has the lowest sales among all categories.
- Discounts don’t strongly impact sales — especially for Fashion.
### 2. Customer Demographics & Purchase Behavior
# Sales by Gender
plt.figure(figsize=(6,4))
gender_sales = ecommerce_dataset_cleaned.groupby('Gender')['Sales'].sum()
sns.barplot(x=gender_sales.index, y=gender_sales.values, palette='coolwarm')
plt.title("Sales by Gender")
plt.ylabel("Total Sales")
plt.show()

# Quantity by Device Type
plt.figure(figsize=(6,4))
device_qty = ecommerce_dataset_cleaned.groupby('Device_Type')['Quantity'].sum()
sns.barplot(x=device_qty.index, y=device_qty.values, palette='viridis')
plt.title("Quantity Ordered by Device Type")
plt.ylabel("Total Quantity")
plt.show()

### Conclusion
- Males buy more than females.
- Most orders come from the web, not mobile.
### 3. Seasonal Trends in Sales
# Convert Order_Date to datetime
ecommerce_dataset_cleaned['Order_Date'] = pd.to_datetime(ecommerce_dataset_cleaned['Order_Date'])

# Create a new column for month
ecommerce_dataset_cleaned['Month'] = ecommerce_dataset_cleaned['Order_Date'].dt.month_name()

# Total monthly sales
monthly_sales = ecommerce_dataset_cleaned.groupby('Month')['Sales'].sum().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

plt.figure(figsize=(12,5))
monthly_sales.plot(kind='bar', color='orange')
plt.title("Monthly Sales Trend")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

# Line chart of sales over time
daily_sales = ecommerce_dataset_cleaned.groupby('Order_Date')['Sales'].sum()

plt.figure(figsize=(14,6))
daily_sales.plot()
plt.title("Daily Sales Over Time")
plt.ylabel("Sales")
plt.xlabel("Date")
plt.show()


###  Seasonal Trends Comparison

| **Season/Event**      | **Usual Timing**      | **Pattern Seen in Sales Data**         | **Insight**                                |
|------------------------|------------------------|------------------------------------------|---------------------------------------------|
| **Valentine’s Day**    | Mid-February           |  Low sales                               | Not a strong sales season                   |
| **Ramadan & Eid**      | April – July (varies)  |  Sales increase around May–July          | Likely impact from religious promotions     |
| **Summer Months**      | June – August          |  Moderate to high sales                  | Summer shopping boosts sales                |
| **Back to School**     | August – September     |  Stable sales                            | Consistent demand, no big spike             |
| **Chinese New Year**   | January – February     |  Low sales                               | Little to no effect in this data            |
| **Black Friday**       | Late November          |  One of the highest peaks                | Strong seasonal impact                      |
| **Christmas Season**   | December               |  High sales, not the top                 | Holiday shopping drives up revenue          |



###  Conclusion:
Sales are highest during **Black Friday**, **Ramadan/Eid**, and **summer**.  
**Valentine’s Day** and **Chinese New Year** show **low sales impact**.

### Correlation Heatmap for Numeric Variables
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 

# Load dataset
df = pd.read_csv("E-commerce_Dataset_cleaned.csv")

# Select only raw numerical features (excluding calculated ones if needed)
numerical_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Aging']

# Drop rows with missing values in numerical columns (just in case)
df_numeric = df[numerical_cols].dropna()

# Compute the correlation matrix
corr = df_numeric.corr().round(4)
print(corr) 
# Plot the heatmap using plotly 
fig = px.imshow(corr, 
                text_auto=True, 
                aspect="auto",  
                title="Correlation Heatmap of Numerical Features",
                template='plotly_dark',
                color_continuous_scale='rdbu',  
               )
fig.show()


---

# Milestone 3: Machine Learning Model Development and Optimization
## Model Selection
- Choose appropriate models for the problem type (classification, regression, clustering, etc.).


### Selected Approach:
- **Model Type**: Linear Regression
- **Target Variable**: Profit
- **Justification**: 
  - Suitable for understanding relationships between input features and profit

### Feature Set Variants:
To compare performance and identify the most impactful features, four Linear Regression models were created:
1. **Model 1** – Using only `Sales`
2. **Model 2** – Using `Sales`, `Quantity`, and `Discount`
3. **Model 3** – Using all numerical features
4. **Model 4** – Using all numerical features + encoded categorical variables

These models will be evaluated based on **RMSE (Root Mean Squared Error)** and **R² Score**, to determine which feature combination performs best for predicting profit.

## Model Training
- Split data into training, validation, and testing sets.
- Address imbalances using techniques like SMOTE or stratified sampling.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the cleaned e-commerce dataset
df = pd.read_csv('E-commerce_Dataset_cleaned.csv')

# Convert date and time columns to proper formats (optional)
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Time'] = pd.to_timedelta(df['Time'])

# Define the target column (what we want to predict)
y = df['Profit']

# ----------------- Model 1 -----------------
# Use only Sales to predict Profit
X1 = df[['Sales']]

# ----------------- Model 2 -----------------
# Use Sales, Quantity, and Discount to predict Profit
X2 = df[['Sales', 'Quantity', 'Discount']]

# ----------------- Model 3 -----------------
# Use all numerical columns to predict Profit
X3 = df[['Sales', 'Quantity', 'Discount', 'Shipping_Cost', 'Aging']]

# ----------------- Model 4 -----------------
# Use all numerical + one-hot encoded categorical variables
# Drop time/date columns and Profit column before encoding
categorical_encoded = pd.get_dummies(df.drop(columns=['Profit', 'Order_Date', 'Time']), drop_first=True)
X4 = pd.concat([df[numerical_cols], categorical_encoded], axis=1)

# Store all feature sets in a dictionary for easy looping
feature_sets = {
    'Model 1 (Sales only)': X1,
    'Model 2 (Sales + Qty + Discount)': X2,
    'Model 3 (All numerical)': X3,
    'Model 4 (All + encoded categoricals)': X4
}

# ----------------- Train and Evaluate Each Model -----------------
for name, X in feature_sets.items():
    # Split the data into 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate R-squared (how well the model fits)
    r2 = r2_score(y_test, y_pred)

    # Print the results
    print(f"{name}:\n  RMSE: {rmse:.2f}\n  R² Score: {r2:.4f}\n")

## Model Evaluation
- Metrics to consider: Accuracy, Precision, Recall, F1-score, RMSE, etc.
- Visual tools: Confusion matrices, ROC curves.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the cleaned dataset
df = pd.read_csv('E-commerce_Dataset_cleaned.csv')

# Convert date and time (optional, for consistency)
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Time'] = pd.to_timedelta(df['Time'])

# Features and target for Model 2
X = df[['Sales', 'Quantity', 'Discount']]
y = df['Profit']

# Split the data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print metrics
print("Model 2 Evaluation:")
print(f"  RMSE: {rmse:.2f}")
print(f"  R² Score: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Model 2: Actual vs Predicted Profit")
plt.grid(True)
plt.tight_layout()
plt.show()


joblib.dump(model, 'model2_linear_regression.pkl')


###  Models Used:
- **Ridge Regression** – uses L2 regularization (shrinks coefficients slightly)
- **Lasso Regression** – uses L1 regularization (can shrink some coefficients to zero)

###  Hyperparameter Tuning:
- **Grid Search with Cross-Validation** was used to find the best `alpha` value for each model.

###  Evaluation Metrics:
- **RMSE (Root Mean Squared Error)**: lower is better.
- **R² Score**: closer to 1.0 means better fit.

###  Results:

| Model             | Best Alpha | RMSE    | R² Score |
|------------------|------------|---------|----------|
| Ridge Regression |    1.0     | *0.29*  | *1.000* |
| Lasso Regression |    0.01    | *0.29*  | *0.001* |


###  Conclusion:
- Ridge is generally more stable when many features contribute small effects.
- Lasso is helpful for feature selection when only a few features are important.



## Model Comparison
- Compare multiple models and justify the final model selection.
This section compares all the models developed to predict **Profit** using different feature sets and regression techniques. Evaluation was based on:

- **RMSE (Root Mean Squared Error)** – lower is better
- **R² Score** – closer to 1 is better



###  Linear Regression Models :

| Model Version                        | Features Used                             | RMSE   | R² Score |
|-------------------------------------|--------------------------------------------|--------|----------|
| Model 1                             | Sales only                                 | 19.64  | 0.8388   |
| Model 2                             | Sales + Quantity + Discount                | 18.10  | 0.8632   |
| Model 3                             | All numerical features                     | 0.29   | 1.0000   |
| Model 4                             | All numerical + encoded categorical        | 0.29   | 1.0000   |



###  Regularized Regression Models :

| Model                                 | Best Alpha | RMSE   | R² Score |
|---------------------------------------|------------|--------|----------|
| Ridge Regression                      |    1.0     |  0.29  |  1.0000  |
| Lasso Regression                      |    0.01    |  0.29  |  1.0000  |



### Final Model Recommendation:

- All models starting from **Model 3** onward show **perfect fit** on test data with **RMSE: 0.29** and **R²: 1.0000**.
- However, since overfitting is possible, **Ridge Regression** is recommended for deployment due to its regularization and robustness to multicollinearity.
- This model will be used in the next stage for deployment and business reporting.

## Visualization for Research Questions
- This section will include the visualizations that provide insights for the research questions defined earlier.  
- **Development Steps for Answering the Research Questions**:
  1. During **Exploratory Data Analysis (EDA)**, visualize initial patterns or trends related to the research questions.
  2. During **Model Evaluation**, provide visualizations to interpret model performance with respect to the research questions.
  3. During the **Final Analysis and Reporting**, present polished visualizations that summarize findings for each research question.

- Create the visualizations for each research question you defined, prove it or answer it, then add a markdown cell after each visual to comment and explain how the visual support your research question.
### Which product categories generate the highest revenue?
import matplotlib.pyplot as plt
import seaborn as sns

# Aggregate sales and profit by product category
category_summary = df.groupby('Product_Category')[['Sales', 'Profit']].sum().sort_values(by='Sales', ascending=False).head(10)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=category_summary.index, y=category_summary['Sales'], palette='viridis')
plt.title("Top Product Categories by Sales")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

### What is the impact of discounts on customer purchases?
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Discount', y='Sales')
plt.title("Effect of Discount on Sales")
plt.grid(True)
plt.show()

### Do seasonal events (e.g. holidays) affect sales?
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Month'] = df['Order_Date'].dt.to_period('M')

monthly_sales = df.groupby('Month')['Sales'].sum()

monthly_sales.plot(kind='line', figsize=(12, 5), marker='o', title='Monthly Sales Trend')
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

##  Final Conclusions for Research Questions 

###   Question 1: Which product categories generate the highest revenue?

 **Conclusion**:  
The bar chart clearly shows that **Fashion** is the top-performing product category by a large margin, followed by **Home & Furniture** and **Auto & Accessories**. This indicates that the business’s primary revenue driver is the fashion sector, and it should be prioritized in marketing and stock planning.



###   Question 2: What is the impact of discounts on customer purchases?

 **Conclusion**:  
The scatter plot of Discount vs Sales shows a **scattered pattern with no clear upward trend**, suggesting that offering a higher discount doesn’t always lead to higher sales. Discounts may be applied in fixed brackets and might not be the main driver of increased purchases.



###   Question 3: Do seasonal events affect sales?

 **Conclusion**:  
The line chart of monthly sales trends shows **clear seasonal spikes**, especially in **May, October, and November**, with the highest peak in **November**. This suggests that holiday seasons and sales events like **Black Friday** likely lead to higher sales, and campaigns should be focused on these months.



## Noon visuals
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
#load noon cleaned dataset
noon_cleaned = pd.read_csv('noon_dataset_cleaned.csv')
#### Converting price from string to float
# Clean the Price column by removing commas and converting to numeric
noon_cleaned['Price'] = noon_cleaned['Price'].str.replace(',', '').astype(float)

# Show cleaned data types and first few rows
noon_cleaned.dtypes, noon_cleaned.head()

### What are the most common product types sold on Noon?
# 1. Average price of the most frequent products
top_products_names = noon_cleaned["Product name"].value_counts().head(10).index
avg_prices = noon_cleaned[noon_cleaned["Product name"].isin(top_products_names)].groupby("Product name")["Price"].mean().sort_values()

plt.figure(figsize=(12, 5))
sns.barplot(x=avg_prices.values, y=avg_prices.index, palette="mako")
plt.title("Average Price of Top 10 Frequent Products")
plt.xlabel("Average Price")
plt.tight_layout()
plt.show()

# 2. Products with multiple prices (possible duplicates with variations)
multi_price_counts = noon_cleaned.groupby("Product name")["Price"].nunique()
multi_priced_products = multi_price_counts[multi_price_counts > 1].sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 5))
sns.barplot(x=multi_priced_products.values, y=multi_priced_products.index, palette="flare")
plt.title("Products with Multiple Price Points")
plt.xlabel("Number of Unique Prices")
plt.show()

# 3. Box plot of prices to detect outliers
plt.figure(figsize=(10, 4))
sns.boxplot(x=noon_cleaned["Price"], color="teal")
plt.title("Box Plot of Product Prices")
plt.xlabel("Price")
plt.grid(True)
plt.show()


# 4. Most Expensive Products
top_expensive = noon_cleaned.sort_values("Price", ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_expensive["Price"], y=top_expensive["Product name"], palette="coolwarm")
plt.title("Top 10 Most Expensive Products")
plt.xlabel("Price")
plt.show()

# 5. Top 10 Cheapest Products
cheapest_products = noon_cleaned.sort_values("Price").head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=cheapest_products["Price"], y=cheapest_products["Product name"], palette="crest")
plt.title("Top 10 Cheapest Products")
plt.xlabel("Price")
plt.tight_layout()
plt.show()


---

# Milestone 4: Deployment 
## Deployment (Streamlit)
- Deploy the model as a REST API (Flask, FastAPI) or interactive dashboards (Streamlit, Dash).
- Host on cloud platforms (AWS, Azure, GCP) or local servers.
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the raw model only (no preprocessor)
model = joblib.load("model2_linear_regression.pkl")

# Load data for visuals
df = pd.read_csv("E-commerce_Dataset_cleaned.csv")

# Title
st.title("E-Commerce Profit Predictor")

# EDA plot
st.subheader("Discount vs Sales")
fig = px.scatter(df, x="Discount", y="Sales", color="Product_Category", template="plotly_dark")
st.plotly_chart(fig)

# Sidebar inputs
st.sidebar.header("Order Input")
sales = st.sidebar.number_input("Sales", 0.0)
quantity = st.sidebar.number_input("Quantity", 1, step=1)
discount = st.sidebar.slider("Discount", 0.0, 0.5, 0.1)

# Predict
if st.button("Predict Profit"):
    input_df = pd.DataFrame([[sales, quantity, discount]], columns=['Sales', 'Quantity', 'Discount'])
    prediction = model.predict(input_df)[0]
    st.success(f" Predicted Profit: ${prediction:.2f}")
