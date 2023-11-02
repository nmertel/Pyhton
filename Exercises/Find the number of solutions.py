'''
Three friends are competing in a quiz competition. 
In order to get a correct answer, at least two of the friends 
must know the answer. Write a Python function that takes as input a 
list of lists of integers, where each integer represents the number of 
friends who know the answer to a particular question. The function should
return the number of questions that the friends can answer correctly.

For example, if the input is:
    
3
1 1 1 
1 0 0
0 1 1

Then the function should return 2, because the friends can answer 2 
questions correctly: question 1, and question 3.
'''
def implementation(n, s):
    count = 0
    for i in range(n):
        views = s[i]
        sure_count = sum(views)
        if sure_count >= 2:
            count += 1    
    return count

# User defined input
num_solutions = int(input("Enter the number of solutions: "))
solutions = []
for j in range(num_solutions):
    solutions.append([int(j) for j in input("Enter the solution: ").split(" ")])

# Output  
result = implementation(num_solutions, solutions)   
print("Result:", result)

