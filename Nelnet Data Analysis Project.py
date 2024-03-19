
# coding: utf-8

# # Nelnet Data Analysis Project

# In[1]:


# Importing necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from tabulate import tabulate
from statsmodels.graphics.tsaplots import plot_acf # Plotting seasonality


# In[2]:


# Loading the dataset

file_path = "C:/Users/nivar/OneDrive/Desktop/approved_data_2016_2018.csv"
df = pd.read_csv(file_path)

# Displaying the first few rows of the dataset to understand its structure
print(df.head())


# In[3]:


# Exploring the dataset

# Checking the dimensions of the dataset
print("Shape of the dataset is:", df.shape)

# Checking the data types and missing values
print(df.info())

# Checking summary statistics for numerical columns
print(df.describe())

# Checking unique values and value counts for categorical columns
df.nunique()


# In[60]:


# Counting the number of null values in each column
null_values = df.isnull().sum()
print(null_values)


# # Data Preprocessing and Data Tidying

# In[62]:


# Dropping unnecessary and irrelevant columns and storing the selected columns in a new dataframe
columns_to_drop = ['emp_title', 'desc', 'mths_since_last_delinq', 'mths_since_last_record', 'last_pymnt_d', 'next_pymnt_d', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'debt_settlement_flag_date', 'settlement_status', 'settlement_date', 'settlement_amount', 'settlement_percentage', 'settlement_term']
new_df = df.drop(columns=columns_to_drop)

# Converting 'emp_length' to string type
new_df['emp_length'] = new_df['emp_length'].astype(str)

# Extracting numeric values from the 'emp_length' column
new_df['emp_length'] = new_df['emp_length'].str.extract('(\d+)').astype(float)

# Converting 'term' column to string type and then removing "months" and converting to numeric
new_df['term'] = new_df['term'].astype(str).str.replace('months', '').astype(int)

# Imputing missing values in 'emp_length', 'dti', and 'revol_util' using median strategy
imputer = SimpleImputer(strategy='median')
new_df[['emp_length', 'dti', 'revol_util']] = imputer.fit_transform(new_df[['emp_length', 'dti', 'revol_util']])
print (new_df.head())

# Substituting empty cells in the 'title' column with "Other"
new_df['title'] = new_df['title'].fillna("Other")

# Removing rows with empty cells in the specified columns
new_df = new_df.dropna(subset=['zip_code', 'inq_last_6mths'])

# Counting the number of null values in each column
null_values = new_df.isnull().sum()
print(null_values)


# In[65]:


# Selecting specific numerical columns
sel_num_col = new_df[['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'installment']]

# Applying describe function to selected numerical columns
sel_num_stat = sel_num_col.describe()

# Transposing the DataFrame for better readability
sel_num_stat = sel_num_stat.T

# Converting DataFrame to a nicely formatted table
table = tabulate(sel_num_stat, headers='keys', tablefmt='github', numalign='left')
print(table)


# In[7]:


# How is loan volume changing over time? Is there any seasonality to their loan origination? 

# Selecting necessary columns
df1 = new_df[['issue_d', 'loan_amnt']]

# Calculating the IQR for 'loan_amnt'
Q1 = df['loan_amnt'].quantile(0.25)
Q3 = df['loan_amnt'].quantile(0.75)
IQR = Q3 - Q1

# Calculating the lower and upper bounds for outliers in 'loan_amnt'
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("Lower bound is:", lower_bound)
print("Upper bound is:", upper_bound)

# Converting date columns to datetime format
df1['issue_d'] = pd.to_datetime(df1['issue_d'])

# Grouping by issue date and calculating loan volume
loan_volume = df1.groupby(df1['issue_d'].dt.to_period('M')).sum()

# Plotting loan volume over time
loan_volume.plot(kind='line', figsize=(10, 6))
plt.xlabel('Issue Date')
plt.ylabel('Loan Volume')
plt.title('Loan Volume Over Time')
plt.show()

# Based on the below graph, the general direction of the loan volume graph (overall trend in loan volume) over time is increasing. 
# So it has positive secular trend and  indicates growing demand for loans. There is no clear seasonality pattern in the loan 
# origination since there isn't a regular gap between all of these peaks and troughs and the time that they are accuring isn't 
# regular. So this variation is cyclic because although it's doing sth over and over and it's got a pattern to it but there isn't
# an even space between each one. So there isn't a probability to when these ups and downs are going to occur.
# There is one significant peak (unusual spike) in loan volume in March 2016 that deviate significantly from the overall trend 
# may indicates periods of increased loan demand, such as during holidays, promotions, or specific economic conditions or 
# outliers.


# In[8]:


# Computing and plotting the autocorrelation function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(loan_volume, lags=35)  # Assuming yearly seasonality, plot ACF up to lag 35
plt.title('Autocorrelation Function (ACF) for Loan Volume')
plt.xlabel('Lag (Months)')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

# In the ACF (autocorrelation function) plot, since we don't observe significant peaks or spikes at regular intervals (lags), it 
# suggests the absence of seasonality or periodic patterns in the data.


# In[10]:


# This company assigns loan grades (A-G) to each loan. Is their grading system indicative of performance? Why/why not?

df2 = new_df[['grade', 'loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti', 'loan_status']]

# calculating summary statistics for each loan grade and performance metric
# Grouping by loan grade and calculating mean performance metrics
grade_performance = df2.groupby('grade').mean()
print(grade_performance) # Mean performance metrics for each loan grade

# The interest rate tends to increase as the loan grade decreases. Grade A loans have the lowest mean interest rate (around 7%),
# while grade G loans have the highest (around 30%).
# The term of the loans also tends to increase as the loan grade decreases. Grade A loans have the shortest mean term (around 37
# months), while grade G loans have the longest (around 54 months).
# There isn't a significant variation in mean employment length across different loan grades. However, there seems to be a slight
# decrease as loan grades go from A to G.
# Annual Income (annual_inc): There's a decreasing trend in mean annual income as loan grades worsen. Grade A loans have the 
# highest mean annual income (around $91,443), while grade G loans have the lowest (around $71,410).
# Debt-to-Income Ratio (dti): The debt-to-income ratio tends to increase as the loan grade decreases. Grade A loans have the 
# lowest mean DTI (around 16.58), while grade G loans have the highest (around 25.09).
# In conclusion, lower loan grades (such as grades D through G) are associated with higher interest rates, longer loan terms, 
# lower annual incomes, and higher debt-to-income ratios compared to higher loan grades (such as grades A and B). This analysis 
# suggests a clear pattern where riskier loan grades come with higher costs and potentially greater financial strain on borrowers.


# In[23]:


# we can visualize the relationship between loan grades and performance metrics using bar plots:

# Defining the desired order of loan grade categories
desired_order = ["A", "B", "C", "D", "E", "F", "G"]

# Reindexing the DataFrame to reorder the rows according to the desired order
grade_performance_ordered = grade_performance.reindex(desired_order)

# Plotting the grouped bar plot
plt.figure(figsize=(12, 6))

# Defining the numerical variables to be plotted
numerical_vars = ['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti']

# Looping through each numerical variable and plotting a grouped bar
for i, var in enumerate(numerical_vars):
    plt.subplot(2, 3, i+1)
    sns.barplot(data=df2, x='grade', y=var, ci=None, palette='viridis', order=desired_order)
    plt.title(f'{var} vs Loan Grade')
    plt.xlabel('Loan Grade')
    plt.ylabel(var)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# We can confirm our result with below visualizations as well that lower loan grades (such as grades D through G) are associated 
# with higher loan amount, longer loan terms, higher interest rates, higher installment, lower annual incomes, and higher dti
# ratios compared to higher loan grades (such as grades A and B)


# In[14]:


# Outside of loan grade, what else would you say are borrower characteristics that affect loan performance?

df3 = new_df[['term', 'int_rate', 'annual_inc', 'loan_status', 'emp_length']]

# calculating summary statistics for each loan status and performance metric
# Groupping by loan status and calculating mean performance metrics
grade_performance = df3.groupby('loan_status').mean()

# Defining the desired order of loan status categories
desired_order = ["Fully Paid", "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)", "Charged Off", "Default"]

# Reindexing the DataFrame to reorder the rows according to the desired order
grade_performance_ordered = grade_performance.reindex(desired_order)
print(grade_performance_ordered) # Mean performance metrics for each loan status


# In[15]:


# Defining the desired order of loan status categories
desired_order = ["Fully Paid", "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)", "Charged Off", "Default"]

# Reindexing the DataFrame to reorder the rows according to the desired order
grade_performance_ordered = grade_performance.reindex(desired_order)

# Plotting the grouped bar plot
plt.figure(figsize=(12, 7))

# Defining the numerical variables to be plotted
numerical_vars = ['int_rate', 'term', 'annual_inc', 'emp_length']

# Preprocessing 'emp_length' column to remove the 'year' suffix and converting to numerical values
df3['emp_length'] = df3['emp_length'].astype(str).str.replace(' years', '').str.replace(' year', '').astype(float)

# Looping through each numerical variable and plotting a grouped bar
for i, var in enumerate(numerical_vars):
    plt.subplot(2, 2, i+1)
    sns.barplot(data=df3, x='loan_status', y=var, ci=None, palette='viridis', order=desired_order)
    plt.title(f'{var} vs Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel(var)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# So, loan status is the other factor that can affect loan performance (interest rate, term, employment length, annual income)
# These results reveal the relationship between loan status and various loan performance metrics, showcasing both positive and 
# negative associations. For instance, lower interest rates and longer employment lengths are generally associated with positive 
# loan outcomes, such as 'Fully Paid' and 'Current' statuses. Additionally, higher annual incomes are typically linked with 
# favorable loan statuses, indicating greater financial stability and repayment capacity. 


# # Exploratory Data Analysis

# In[20]:


# Loan Amount and Interest Rate Distributions

# Setting seaborn style to "darkgrid"
sns.set_style("darkgrid")

# Creating a figure and two subplots arranged horizontally
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plotting the distribution of loan amounts
sns.distplot(new_df.loan_amnt, ax=axes[0])
axes[0].set_title("Distribution of Loan Amount")

# Plotting the distribution of interest rates
sns.distplot(new_df.int_rate, ax=axes[1])
axes[1].set_title("Distribution of Interest Rate")

# Adjusting layout
plt.tight_layout()
plt.show()

# The loan amount graph exhibits a prominent peak at $10,000, followed by a gradual decline in the curve. This pattern implies a 
# lower preference among investors for larger loan amounts, with a predominant focus on smaller to mid-range loan investments.
# In the loan interest distribution, a notable concentration of investors is observed within the 10 to 15% interest rate range. 
# Beyond this range, there is a gradual decrease in the number of investors, indicating a preference for lower interest rates 
# among the majority of borrowers.


# In[83]:


# Loan Status Distribution

# Groupping loans by their status and counting the number of loans in each status category
loan_status = new_df.groupby('loan_status')['loan_status'].count()

# Generating the pie chart with loan status counts
plt.figure(figsize=(10,10))
plt.title('Loans Status', fontsize=15)
pie = plt.pie(loan_status, autopct='%1.1f%%',labels=loan_status.index ,startangle=180)
# Making the numbers inside the pies bold
for text in pie[1]:
    text.set_fontweight('bold')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
# Making the labels bold
plt.setp(pie[1], fontweight='bold')
loan_status

# The distribution suggests that a significant number of loans are still in progress ("Current"), while a notable portion has 
# been fully paid. However, there are also instances of delinquency ("Late (16-30 days)", "Late (31-120 days)", "In Grace Period")
# and a small number of defaults. This indicates successful investment outcomes for a significant portion of investors.


# In[59]:


# Loan Distributions by Purposes and States

# Loan Purpose Distribution
loan_purpose_counts = new_df['purpose'].value_counts()
plt.figure(figsize=(18, 6))

# Plotting Loan Purpose Distribution
plt.subplot(1, 2, 1)
loan_purpose_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Loan Purposes', fontsize=15)
plt.xlabel('Loan Purpose', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(rotation=45, ha='right', fontsize=12)  # Increasing font size for x-axis ticks
plt.yticks(fontsize=12)  # Increasing font size for y-axis ticks

# Geographic Distribution Analysis
loan_counts_by_state = new_df['addr_state'].value_counts()

# Plotting Geographic Distribution Analysis
plt.subplot(1, 2, 2)
loan_counts_by_state.plot(kind='bar', color='lightgreen')
plt.title('Loan Distribution by State', fontsize=15)
plt.xlabel('State', fontsize=15)
plt.ylabel('Number of Loans', fontsize=15)
plt.xticks(rotation=90, fontsize=11)  # Increasing font size for x-axis ticks and rotate labels
plt.yticks(fontsize=12)  # Increasing font size for y-axis ticks
plt.tight_layout()  # Adjusting layout to prevent overlap
plt.show()

# The frequency distribution of loan purposes was calculated, revealing that the most common reasons borrowers seek loans are 
# debt consolidation and credit card, respectively.
# I explored the geographic distribution of loan originations to identify regions with the highest loan activity. The states with
# the highest loan activity are California, Texas, New York, and Florida, respectively.


# In[92]:


# Calculating correlation matrix
corr_matrix = df2.corr()

# Plotting correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Features')
plt.show()

# A strong correlation is observed between installment vs loan amount.
# Correlation heatmap helps to visualize the relationships between variables, detecting pattern and feature selection by removing
# highly correlated variables.

