from matplotlib import pyplot as plt

print(plt.style.available)
plt.style.use('bmh')

# The data:
## Median Developer Salaries by Age
ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]

# Let's plot
## Generate the plot
plt.plot(ages_x,
         dev_y,
         marker='.',
         label='All Devs')


# Median Python Developer Salaries by Age
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

plt.plot(ages_x,
         py_dev_y,
         marker='o',
         label='Python')

plt.title('Median salary (USD) by age')
plt.xlabel('Ages')
plt.ylabel('Median salary (EUR)')

## Create legend, passing a list of names in the same order in which variables were added.
#plt.legend(['All Devs', 'Python'])
plt.legend()
## There is a better way to do this.

## Add a grid
#plt.grid()

## Manage padding:
plt.tight_layout()

# Save the plot
#plt.savefig('plot.png')

## Show the plot
#plt.show()
print("\n\n\n\n")
print(globals())

