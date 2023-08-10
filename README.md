Titanic Dataset Analysis

Overview
This project focuses on downloading, analyzing, and visualizing the famous Titanic dataset. This dataset contains information about the passengers aboard the RMS Titanic, which tragically sank on its maiden voyage. It includes data such as age, fare, class, and whether the passenger survived or not.

Workflow

1. DATA ACQUISITION
Using the Pandas library, the dataset is directly downloaded from Stanford University's CS109 course archives.

                    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
                    data = pd.read_csv(url)

2. DATA OVERVIEW
The initial lines of the dataset are displayed to understand its structure and columns. We also extract information about the number of entries, column data types, and any missing values.

                    print(data.head())
                    print(data.info())

3. SUMMARY STATISTICS
Descriptive statistics are computed to get a sense of data distribution and central tendencies. 

                    print(data.describe())

4. MISSING DATA CHECK
The dataset is inspected for any missing data, allowing us to address them in subsequent analyses if needed.

                    print(data.isnull().sum())

5. DATA VISUALIZATION
Several visualizations are plotted using the Seaborn library to explore relationships and patterns in the data:

Survival Count: Displays the number of passengers who survived versus those who didn't.
Survival by Class: Indicates the survival rate across different passenger classes.
Age Distribution: Illustrates the age distribution of the passengers.
Fare Distribution: Displays how much passengers paid for their tickets.
Age Distribution by Class: Shows the age distribution within each passenger class.
Correlation Heatmap: Visualizes the correlation between numeric columns in the dataset.
6. DATA EXPORT
The dataset is then saved locally for future analyses or to be used by other programs.

                   data.to_csv("titanic_dataset.csv", index=False)

Conclusion
This project offers a foundational analysis of the Titanic dataset, providing insights into the tragic event's demographics and outcomes. The visualizations give a clear understanding of how different factors might have influenced the chances of survival. This dataset serves as a starting point for numerous machine learning and statistical modeling tasks.
