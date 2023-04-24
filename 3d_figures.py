import matplotlib.pyplot as plt
from scipy.optimize import fmin
import numpy as np
import pygad
from matplotlib.colors import LinearSegmentedColormap

def F (x):
    z = x[0] + 1j * x[1]
    return 1/(1 + abs(z ** 6 - 1))

def F_inv(x):
    return -1 * F(x)




def makeData ():

    x = np.linspace (-2, 2, 1000)
    y = np.linspace (-2, 2, 1000)

    # Создаем двумерную матрицу-сетку
    xgrid, ygrid = np.meshgrid(x, y)
    z_comp = xgrid + 1j * ygrid
    nums = 1 / (1 + abs(z_comp**6 - 1))

    return xgrid, ygrid, nums

def Bruteforce():
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    xgrid, ygrid = np.meshgrid(x, y)
    z_comp = xgrid + 1j * ygrid
    zgrid = 1 / (1 + abs(z_comp ** 6 - 1))
    all_sorted = {}
    for a in range (len(xgrid)):
        for b in range (len(xgrid[0])):
            i = a
            j = b
            all_sorted[xgrid[i, j], ygrid[i, j]] = zgrid[i, j]

    all_sorted = dict(sorted(all_sorted.items(), key=lambda item: item[1]))
    max_sorted = {}
    for a in range (1, len(xgrid) - 1):
        for b in range (1, len(xgrid[0]) - 1):
            val = all_sorted[xgrid[a, b], ygrid[a, b]]
            val_left = all_sorted[xgrid[a - 1, b], ygrid[a - 1, b]]
            val_right = all_sorted[xgrid[a + 1, b], ygrid[a + 1, b]]
            val_up = all_sorted[xgrid[a, b + 1], ygrid[a, b + 1]]
            val_down = all_sorted[xgrid[a, b - 1], ygrid[a, b - 1]]
            if val > val_down and val > val_up and val > val_right and val > val_left:
                max_sorted[xgrid[a, b], ygrid[a, b]] = val



    return max_sorted




def fitness_func(ga_instance, solution, solution_idx):
    z = solution[0] + 1j * solution[1]
    return 1 / (1 + abs(z ** 6 - 1))


fitness_function = fitness_func

num_generations = 50
num_parents_mating = 4

sol_per_pop = 8
num_genes = 2 #len(function_inputs)

init_range_low = -2
init_range_high = 2

parent_selection_type = "sss" #steady state selection
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10
all_solutions = {}
for i in range (100):
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if i == 0:
        all_solutions[solution[0], solution[1]] = solution_fitness
        continue

    break_flag = 0
    for pair in all_solutions:
        x1 = solution[0]
        y1 = solution[1]
        x2 = pair[0]
        y2 = pair[1]
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if dist < 0.2:
            break_flag = 1
            break
    if break_flag == 1:
        continue

    all_solutions[solution[0], solution[1]] = solution_fitness

for pair in all_solutions:
    print(pair, all_solutions[pair])



#print("Parameters of the best solution : {solution}".format(solution=solution))
#print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#prediction = F(solution)  #np.sum(np.array(function_inputs)*solution)
#print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

#fmin(F_inv, np.array([-2, -2]))

"""
brute = Bruteforce()
for pair in brute:
    print(pair, brute[pair])

x, y, z = makeData()

fig = plt.figure()
axes = fig.add_subplot(projection='3d')


cmap = LinearSegmentedColormap.from_list ('red_blue', ['b', 'w', 'r'], 256)
axes.plot_surface(x, y, z, color='#11aa55', cmap=cmap)
for pair in brute:
    x_r = round(pair[0], 2)
    y_r = round(pair[1], 2)
    z_r = round(brute[pair], 2)
    axes.scatter(pair[0], pair[1], brute[pair], c='g')
    axes.text(x_r, y_r, z_r, f"({x_r}, {y_r}, {z_r})")
    #plt.annotate(f"({x_r}, {y_r}, {z_r})", (x_r, y_r, z_r))




plt.show()
"""