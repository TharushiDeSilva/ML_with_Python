from pprint import PrettyPrinter
import pandas as pd 
from tabulate import tabulate

# simple dataset of people 
data = {'Name': ["John", "Anna", "Peter", "Linda"], 
    'Location': ["New York", "Paris", "Berlin", "London"], 
    'Age': [24, 13, 53, 33]}

data_pandas = pd.DataFrame(data)
print(data_pandas.to_markdown())

# we can query the above table 
print()
age_30 = data_pandas[data_pandas.Age > 30]
print(age_30.to_markdown())