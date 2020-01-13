print('Running...')
#get data
csv_file = 'auto_insurance_sweden.csv'

x=[]
with open(csv_file, 'r') as f:
    for row in f:
        x.append(row.split(',')[0])
x = [float(i) for i in x]

y=[]
with open(csv_file, 'r') as f:
    for row in f:
        y.append(row.split(',')[1])
y = [float(i) for i in y]

###########################################################################################

def computeError(x,y,slope,yint):
    errorlist = [abs(y[i]-(slope*x[i]+yint)) for i in range(len(x))] #(y-(mx+b))^2
    error = sum(errorlist)/len(errorlist) #take average
    return error

def gradientDescent(x,y,initial_slope,initial_yint,learningrate,epochs):
    slope = initial_slope
    yint = initial_yint
    for i in range(epochs):
        slopegradient = (-2/len(x)) * sum([x[i]*(y[i]-(slope*x[i]+yint)) for i in range(len(x))]) #partial derivative (of error function) with respect to slope
        yintgradient = (-2/len(x)) * sum([y[i]-(slope*x[i]+yint) for i in range(len(x))]) #partial derivative (of error function) with respect to y intercept
        slope = slope - (learningrate*slopegradient) #update slope
        yint = slope - (learningrate*yintgradient) #update y intercept
    return slope,yint

learningrate = 0.0001
epochs = 100000
initial_slope = 0
initial_yint = 0
[slope, yint] = gradientDescent(x,y,initial_slope,initial_yint,learningrate,epochs)
error = computeError(x,y,yint,slope)
print('Error: '+str(error))

###########################################################################################

import matplotlib.pyplot as plt
import numpy as np

#graph points and line of best fit
print('y = '+str(slope)+'*x + '+str(yint))
xline = np.linspace(len(x),int(min(x)),int(max(x)))
yline = slope*xline+yint
plt.scatter(x, y) #plot the original points
plt.plot(xline,yline,'-r', label='y='+str(slope)+'*x+'+str(yint))
plt.show()
