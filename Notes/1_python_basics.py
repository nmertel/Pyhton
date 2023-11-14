# PYTHON REMINDER
# %% Spyder Introduction
"""
Integrated Development Environment, IDE
Why I prefer: 
    Anaconda Distribution 
    Variable Explorer: libraries and IDEs
Alternative:
    Pycharm
    Jupyter notebook, Jupyter Lab
    Google colab
"""
# %% Variables
"""
 - 4 types of basic variables: int, str, double, bool
 - basic mathematical operations
 - print function
 - conversion between variables
"""
integer_variable = 10
decimal_variable = 12.3

# 4 basic operations
pi_number = 3.14
coefficient = 2
sum_val = pi_number + 1 # sum
difference = pi_number - 1 # subtraction
multiplication = pi_number * coefficient # multiplication
division = pi_number / coefficient # division

# print function
print("Sum: ", sum_val) # print function is used to display variables, use paranthesis to define functions
print("Difference: {}".format(difference)) # use curly braces
print("Multiplication: %.1f, Division: %.4f" % (multiplication, division)) # percent sign with a letter f

# variable type conversion
multiplication_int = int(multiplication) 
print(multiplication_int)
integer_float = float(integer_variable)
print(integer_float)

# string: sequence of characters
string_var = "hello world"
print(string_var)

boolean = True
print(boolean)
# %% 
# Python Basic Syntax
"""
We will learn syntax such as 
    uppercase/lowercase distinction, 
    commenting, 
    code indentation, 
    keywords.
"""
basic = 6
BASIC = 7
print(basic)
print(BASIC)

# Writing my first comment here. # = number sign
"""
Ben python ogreniyorum
"""

# We will learn the codes written in this section in the upcoming parts.
# An example to illustrate the importance of indentation
# in Python you need to use indentation in if-else statements, functions, loops
if 5<10:
    print("5 is less than 10")

var = 4
#def = 4:

#1num = 4
#num1 = 4
# %% list
"""
* Lists are used for store different types of variables
* A list contains items separated by commas and enclosed within square brackets ([]).
* Lists are somewhat similar to arrays in C or java.
* One of the differences between them is that all the items belonging to a list can be of different data types.
In list: 
    create list
    indexing
    add variable into list
    remove variable from list
    reverse
    sort
"""
my_list = [1,2,3,4,5,6] # use square brackets to define lists 
print(type(my_list))

# first element of the list
# index starts from zero in python
week = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
print(week[0])

# last element of the list
print(week[-1])

# number of elements in the list
print(len(week))

# second, third, and fourth element of the list
print(week[2:5]) # week[x,y]: x = inclusive, y = exclusive

number_list = [1,3,2,4,6,5]
number_list.append(7) # adds an element to the end
print(number_list)

number_list = [1,3,2,4,6,5]
number_list.remove(4) # removes the specified element
print(number_list)

number_list = [1,3,2,4,6,5]
number_list.reverse() # reverses the list
print(number_list)

number_list = [1,3,2,4,6,5]
number_list.sort() # sorts the list in ascending order
print(number_list)
# %% Tuple
"""
* Tuples are one of the collections in Python, which are ordered and immutable.
* They are written using parentheses ( ).
* Tuples can be considered as read-only lists.
"""
tuple_datatype = (1,2,3,3,4,5,6)
print(tuple_datatype[0]) # first element = zeroth index

tuple_datatype = (1,2,3,3,4,5,6)
print(tuple_datatype[2:]) # print elements from the 2nd index onward

tuple_datatype = (1,2,3,3,4,5,6)
print(tuple_datatype.count(3)) # counts how many times 3 appears

tuple_xyz = (1, 2, 3)
x,y,z = tuple_xyz
print(x,y,z)
# %% Dictionary
"""
* Python dictionaries are a type of hash table.
* They consist of key-value pairs.
* Dictionary keys can be almost any Python type, but are generally numbers or strings.
* On the other hand, values can be any arbitrary Python object.
* For example: {"Key": value}
"""
cities = {"Istanbul":34, "Izmir":35, "Konya":42}
print(cities)

print(cities['Istanbul'])  # Value for the key 'Istanbul'

print(cities.keys())  # All keys

print(cities.values())  # All values

# %% if - else
"""
* Conditional statements and structures allow different calculations or actions to be performed based on the evaluation 
of a boolean condition specified by the programmer as either true or false.
"""
# Comparing greater or lesser
num1 = 12.0
num2 = 20.0
if num1 < num2:
    print("num1 is less than num2")
elif num1 > num2:
    print("num1 is greater than num2")
else:
    print("num1 is equal to num2")

numbers = [1,2,3,4,5]
value = 31
if value in numbers:
    print("{} is in the list".format(value))
else:
    print("{} is NOT in the list".format(value))
    
capitals = {"Turkey":"Ankara", "England":"London", "Spain":"Madrid"}
keys = capitals.keys()
value = "Turkey"
if value in keys:
    print("{} is in the dictionary".format(value))
else:
    print("{} is NOT in the dictionary".format(value))
   
bool1 = True 
bool2 = False
if bool1 or bool2: # and
    print("success")
else:
    print("failure")

# %% Loops
"""
* The for loop is used to iterate over a sequence. This sequence can be a list, a dictionary, or a string.
"""

# for
# range writes values from 1 to 11 to the parameter i in each iteration
for i in range(1, 11):
    print(i)
    
numbers = [1, 4, 5, 6, 8, 3, 3, 4, 67]
# Addition with for loop
total = 0
for num in numbers:
    total += num
print(total)
 
tuples = ((1, 2, 3), (3, 4, 5))
for x, y, z in tuples:
    print(x + y + z)
    
# while 
i = 0
while i < 4:
    print(i)
    i += 1    
    
# Addition with while
numbers_list = [1, 4, 5, 6, 8, 3, 3, 4, 67]

limit = len(numbers_list)   
counter = 0
calculation = 0
while counter < limit:
    calculation += numbers_list[counter]
    counter += 1 
print(calculation)

# %% Functions
"""
* The main task of functions is to bring complex operations together so that we can perform these operations in a single step.
* Functions often act as a template for the operations we want to perform.
* Using functions, we can gather operations consisting of one or several steps under a single name. 

lambda function
"""

# user-defined function

# def = definition
# circleArea = function name
# r = input parameter
def circleArea(r):
    """
        Calculates the area of a circle
        Input = Circle Radius
        Output = Circle Area
    """
    pi = 3.14
    area = pi * (r**2)
    return area
area = circleArea(3)
print(area)

def circleCircumference(r, pi=3.14):
    """
        Calculates the circumference of a circle
        Input = Circle Radius
        Output = Circle Circumference
    """
    circumference = 2 * pi * r
    print(circumference)
circleCircumference(3)

multiplier = 5
def multiplierMultiplication():
    global multiplier
    print(multiplier * multiplier)
multiplierMultiplication()

def empty():
    pass

# built-in functions: 
numbers = [1, 2, 3, 4]

print(len(numbers))
print(str(numbers))
numbers_copy = numbers.copy()
print(numbers_copy) 
print(max(numbers))
print(min(numbers))

# %% numpy
"""
NumPy is a library for the Python programming language that provides computing convenience for large, multi-dimensional arrays and matrices while containing a wide collection of high-level and complex mathematical functions necessary to work with these arrays.
"""
import numpy as np

# Let's create an array with the dimension of 1*15
array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(array)

# Let's look at the dimension of the array
print(array.shape)

# Let's transform the 1*15 array to a 3*5 matrix
array2 = array.reshape(3,5)

print("Shape: ",array2.shape)
print("Dimension: ", array2.ndim) # 2 dimensional array

print("Data Type: ",array2.dtype.name) # data type inside the array - integer
print("Size: ",array2.size) # size of the array

# type of the array
print("type: ",type(array2))

# Let's create a 2 dimensional array
array2D = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,5]])
print(array2D)

# Let's create an array consisting of zeros
zero_array = np.zeros((3,4))
print(zero_array)

# Let's create an array consisting of ones
one_array = np.ones((3,4))
print(one_array)

# Let's create an empty array
empty_array = np.empty((2,3))
print(empty_array) # in python, zero means empty. 
# As seen below, the numbers are approaching zero

# arange(x,y, step): starts from x, goes until y (excluding y), increasing by step size
array_range = np.arange(10,50,5)
print(array_range)

# linspace(x, y, step) divides between x and y (including x and y) into step number of parts
array_space = np.linspace(10,20,5)
print(array_space)

float_array = np.float32([[1,2],[3,4]])
print(float_array)

# For mathematical operations, let's create 2 arrays
a = np.array([1,2,3])
b = np.array([4,5,6])

print(a+b) # array addition
print(a-b) # array subtraction
print(a**2) # multiplication of array with itself

# sum of elements in the array
print(np.sum(a))

# finding the largest value of the array
print(np.max(a))

# finding the smallest value of the array
print(np.min(a))

# average of the array
print(np.mean(a))

# median of the array
print(np.median(a))
# generate a random number between [0,1] Continuous uniform distribution - 3*3 matrix
random_array = np.random.random((3,3))
print(random_array)

array = np.array([1,2,3,4,5,6,7])
# the first element of the array, which is the element at the zeroth index
print(array[0])

# the first 4 elements of the array
print(array[0:4])

# the reverse of the array
print(array[::-1])

# to examine indices and slices in matrices, let's create a 2D matrix
array2D = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(array2D)

# the element at the 1st row and 1st column of the array, remember we start counting from zero
print(array2D[1,1])

# all rows of the 1st column of the array
print(array2D[:,1])

# 1st, 2nd, 3rd element of the 1st row of the array
print(array2D[1,1:4])

# all columns of the last row of the array
print(array2D[-1,:])

# all rows of the last column of the array
print(array2D[:,-1])

array2D = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array2D)

# convert to vector
vector = array2D.ravel()
print(vector)

index_of_maximum_number = vector.argmax()
print(index_of_maximum_number)
print(vector.reshape(3,3))

# %%
"""
* Pandas is a software library written for data manipulation and analysis in Python. 
This library primarily creates a data structure for processing time-labeled series and numerical tables, 
and various operations can be performed on this data structure.
* With Pandas, data analysis and processing can be done quickly, powerfully, and flexibly.
* Thanks to Pandas, we can easily read .csv and .txt files.
* In data science, the Pandas library facilitates finding and removing missing data in the data.
"""
# Import the Pandas library
import pandas as pd

# To create a pandas dataframe, you first need to create a dictionary
dictionary = {"name": ["Alice","Bob","Charlie","David","Eva","Fiona"],
              "age" : [15,16,17,33,45,66],
              "salary": [100,150,240,350,110,220]} 
data = pd.DataFrame(dictionary)
print(data)

# For a first impression on the data
print(data.head())
print(data.columns) # columns of the data
print(data.info())
print(data.describe())

# Get the age column
print(data["age"])

# Add a new column to the data with world capitals
data["city"] = ["London","Paris","Berlin","Tokyo","Moscow","Washington"]
print(data)

# Get the age column using another method
print(data.loc[:,"age"])

# Get the age column and the first 3 rows, including the third row
print(data.loc[:3,"age"])

# Get the first 3 rows and columns from age to city
print(data.loc[:3,"age":"city"])

# Get the first 3 rows and the name and age columns
print(data.loc[:3,["name","age"]])

# Print rows in reverse
print(data.loc[::-1,:])

# Print the age column using iloc. The age column is at index 1 in the list
print(data.iloc[:,1])

# Get the first 3 rows and the name and age columns using iloc
print(data.iloc[:3,[0,1]])

# Filtering
dictionary = {"name": ["Alice","Bob","Charlie","David","Eva","Fiona"],
              "age" : [15,16,17,33,45,66],
              "city": ["London","Paris","Berlin","Tokyo","Moscow","Washington"]} 
data = pd.DataFrame(dictionary)
print(data)

# First, let's create a filter based on age. age > 22
filter1 = data.age > 22
filtered_data = data[filter1]
print(filtered_data)

# list comprehension
# Find the average age
average_age = data.age.mean()
# We could also use np.mean(data.age)
print(average_age)

data["AGE_GROUP"] = ["young" if average_age > i else "old" for i in data.age]
print(data)

# Merging
dict1 = {"name": ["Alice","Bob","Charlie"],
              "age" : [15,16,17],
              "city": ["London","Paris","Berlin"]} 
data1 = pd.DataFrame(dict1)

# Create dataset 2
dict2 = {"name": ["David","Eva","Fiona"],
              "age" : [33,45,66],
              "city": ["Tokyo","Moscow","Washington"]} 
data2 = pd.DataFrame(dict2)

# Vertical merge, if axis=0 then it's vertical
data_vertical = pd.concat([data1,data2],axis=0)

# Horizontal merge, if axis=1 then it's vertical
data_horizontal = pd.concat([data1,data2],axis=1)

# %% Visualization with matplotlib
"""
Data visualization is the graphical representation of information and data.
* Using visual elements like charts, graphs, and maps, data visualization tools provide 
an accessible way to see and understand trends, outliers, and patterns in the data.
"""

import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4]
y = [4,3,2,1]
plt.figure()  # Create a figure - outer frame
plt.plot(x, y, color="red", alpha=0.7, label="line")  # alpha = transparency
plt.scatter(x, y, color="blue", alpha=0.7, label="scatter")  # alpha = transparency
plt.title('Matplotlib')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.xticks([0,1,2,3,4,5])
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(9, 7))
fig.subplots_adjust(hspace=0.5)

x = [1,2,3,4,5,6,7,8,9,10]
y = [10,9,8,7,6,5,4,3,2,1]

axes[0].scatter(x, y)
axes[0].set_title("Subplot 1")
axes[0].set_ylabel("y values")
axes[0].set_xlabel("x values")

axes[1].scatter(y, x)
axes[1].set_title("Subplot 2")
axes[1].set_ylabel("y values")
axes[1].set_xlabel("x values")

plt.figure()
img = np.random.random((50,50))
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# %% Operating System (OS) module in Python
import os  # The OS module in Python provides functions for interacting with the operating system

print(os.name)  # It returns 'nt' for Windows, and 'posix' for Linux.

current_dir = os.getcwd()
print(current_dir)  # This method returns the current directory we are in.

folder_name = "new_folder"
os.mkdir(folder_name)

new_folder_name = "new_folder2"
os.rename(folder_name, new_folder_name)

os.chdir(current_dir + "\\" + new_folder_name)  # This lets you change to a different directory.
print(os.getcwd())

os.chdir(current_dir)
print(os.listdir())  # This returns the files and folders in the current directory.

files = os.listdir()
for f in files:
    if f.endswith(".py"):
        print(f)

os.rmdir(new_folder_name)  # This is used to remove an empty directory. Let's delete the 'new_folder2' directory we just created.

# This method allows you to see the directories and files under a given directory.
for dirpath, dirnames, filenames in os.walk(current_dir):
    print(dirpath, dirnames, filenames)

os.path.exists("1_python_basics.py")











