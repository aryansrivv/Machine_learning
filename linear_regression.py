from numpy import *
# collecting data 
points = genfromtxt('data.csv' , delimiter = ',')


def compute_error_for_line_given_points(b,m,points):
    #initial error at 0 
    total_error = 0 
    for i in range(len(points)):
        x = points [i,0] # get the x value 
        y = points[i,1]  # get the y value 
        
        # get difference 
        
        total_error += (y-(m*x+b))**2
        
    return total_error/float(len(points))



def gradient_decent_runner(points , starting_b , starting_m , learning_rate , num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b,m = step_gradient(b,m,array(points), learning_rate)
    return [b,m]


def step_gradient(b_current , m_current , points , learning_rate):
    # initializing b and m (partial gradient)
    b_gradient = 0 # -2/N sigma (i = 1 to N) ( y(i) - m*x(i) - b )
    m_gradient = 0 # -2/N sigma (i = 1 to N) ( y(i) - m*x(i) - b )*(x(i))
    
    # here N = 100 ( len(points))
    
    n = float(len(points))
    
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        # computing partial derivatives of error function 
        b_gradient += -(2/n)*(y-((m_current*x)+b_current))
        m_gradient += -(2/n)*(y-((m_current*x)+b_current))*x
        
        
    # update b and w 
    new_b = b_current - (learning_rate*b_gradient)
    new_m = m_current - (learning_rate*m_gradient)
    return [new_b,new_m]
        
        
        
# Step 2 - defining hyperparameters 

learning_rate = 0.0001  #how fast should our model converge ?

# y = wx + b 

initial_m = 0 
initial_b = 0 
num_iterations = 1000

# train our model 

print(f'starting gradient decent at b = {initial_b} , w = {initial_m} , error = {compute_error_for_line_given_points(initial_b , initial_m, points)} ')


[b,m] = gradient_decent_runner(points, initial_b, initial_m,learning_rate,num_iterations)


print(f'ending point at b = {b} , w = {m} , error = {compute_error_for_line_given_points(b , m, points)} ')