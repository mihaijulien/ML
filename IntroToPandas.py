import pandas as pd
import numpy as np
print(pd.__version__)
print(np.__version__)

# The primary data structures in pandas are implemented as two classes:
#
# DataFrame, which you can imagine as a relational data table, with rows and named columns.
# Series, which is a single column. A DataFrame contains one or more Series and a name for each Series.

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

pd.DataFrame({ 'City name': city_names, 'Population': population })

# Most of the time, you load an entire file into a DataFrame
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
# DataFrame.describe is used to show interesting statistics about a DataFrame
california_housing_dataframe.describe()
# DataFrame.head displays the first few records of a DataFrame
print(california_housing_dataframe.head())

# Accessing Data

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(type(cities['City name']))  # <class 'pandas.core.series.Series'>
print(cities['City name'])

print(type(cities['City name'][1])) # <class 'str'>
print(cities['City name'][1]) # San Jose

print(type(cities[0:2])) # <class 'pandas.core.frame.DataFrame'>
print(cities[0:2])

# The following code adds two Series to an existing DataFrame:
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print(cities)

#Note: Boolean Series are combined using the bitwise, rather than the traditional boolean, operators. For example, when performing logical and, use & instead of and.
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
cities


# UltraQuick Pandas tutorial

# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)

# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])

# Create a Python list that holds the names of the four columns.
my_column_names = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

# Create a 3x4 numpy array, each cell populated with a random integer.
my_data = np.random.randint(low=0, high=101, size=(3, 4))

# Create a DataFrame.
df = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(df)
