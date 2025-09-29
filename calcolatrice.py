import matplotlib.pyplot as plt
import time



# The computed tour is[2, 1, 3, 0]

myList = [10, 20, 30, 40, 50]
print("Original:", myList)

# Shift down, remove last, prepend None
myList = [None] + myList[:-1]
print("Shifted down:", myList)



tour = [1,2,3,4,10,9,8,7,6,5,11,12,13,14]
print(tour)
i = 3
j = 9
newTour = []
newTour.extend(tour[0:i+1])
newTour.extend(reversed(tour[i+1:j+1]))
newTour.extend(tour[j+1:])
tour = newTour
print(tour)

# myList = [10, 20, 30, 40, 50]
# selection = myList[2:-1]
# print(selection)

# Activate interactive mode
plt.ion()

fig, ax = plt.subplots()
x_values, y_values = [], []
line, = ax.plot([], [], 'b-')

for i in range(1, 101):
    y = i**2
    x_values.append(i)
    y_values.append(y)

    line.set_xdata(x_values)
    line.set_ydata(y_values)

    ax.relim()
    ax.autoscale()

    plt.draw()
    plt.pause(0.01)   # very short pause for smooth updates

plt.ioff()
plt.show()
