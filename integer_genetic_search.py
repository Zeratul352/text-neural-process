import numpy as np
import pygad


def fitness_func(ga_instance, solution, solution_idx):
    x = solution[0]
    y = solution[1]
    if 25 * x + 500 * y > 1000:
        return 0
    if 2 * y > x:
        return 0
    return x + 25 * y

fitness_function = fitness_func

num_generations = 200
num_parents_mating = 4

sol_per_pop = 8
num_genes = 2 #len(function_inputs)

init_range_low = 0
init_range_high = 50

parent_selection_type = "sss" #steady state selection
#rank_selection  #random_selection
#wheel_selection  #tournament_selection
keep_parents = 1

crossover_type = "single_point"  #two_points_crossover
#uniform_crossover  #scattered_crossover
mutation_type = "random"
#swap_mutation  #inversion_mutation
#scramble_mutation  #adaptive_mutation
mutation_percent_genes = 10

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
                           mutation_percent_genes=mutation_percent_genes,
                           gene_type=int)

ga_instance.run()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution, solution_fitness)