import matplotlib.pyplot as plt
from scipy.optimize import fmin
import numpy as np
import time
import pygad
from matplotlib.colors import LinearSegmentedColormap
#from sko.AFSA import AFSA
from scipy import spatial

def F(x):
    z = x[0] + 1j * x[1]
    return 1 / (1 + abs(z ** 6 - 1))


def F_inv(x):
    return -1 * F(x)


def makeData():
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)

    # Создаем двумерную матрицу-сетку
    xgrid, ygrid = np.meshgrid(x, y)
    z_comp = xgrid + 1j * ygrid
    nums = 1 / (1 + abs(z_comp ** 6 - 1))

    return xgrid, ygrid, nums


def Bruteforce():
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    xgrid, ygrid = np.meshgrid(x, y)
    z_comp = xgrid + 1j * ygrid
    zgrid = 1 / (1 + abs(z_comp ** 6 - 1))
    all_sorted = {}
    for a in range(len(xgrid)):
        for b in range(len(xgrid[0])):
            i = a
            j = b
            all_sorted[xgrid[i, j], ygrid[i, j]] = zgrid[i, j]

    all_sorted = dict(sorted(all_sorted.items(), key=lambda item: item[1]))
    max_sorted = {}
    for a in range(1, len(xgrid) - 1):
        for b in range(1, len(xgrid[0]) - 1):
            val = all_sorted[xgrid[a, b], ygrid[a, b]]
            val_left = all_sorted[xgrid[a - 1, b], ygrid[a - 1, b]]
            val_right = all_sorted[xgrid[a + 1, b], ygrid[a + 1, b]]
            val_up = all_sorted[xgrid[a, b + 1], ygrid[a, b + 1]]
            val_down = all_sorted[xgrid[a, b - 1], ygrid[a, b - 1]]
            if val > val_down and val > val_up and val > val_right and val > val_left:
                max_sorted[xgrid[a, b], ygrid[a, b]] = val

    return max_sorted

class AFSA:
    def __init__(self, func, n_dim, size_pop=50, max_iter=300,
                 max_try_num=100, step=0.5, visual=0.3,
                 q=0.98, delta=0.5):
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.max_try_num = max_try_num  # 最大尝试捕食次数
        self.step = step  # 每一步的最大位移比例
        self.visual = visual  # 鱼的最大感知范围
        self.q = q  # 鱼的感知范围衰减系数
        self.delta = delta  # 拥挤度阈值，越大越容易聚群和追尾

        self.X = (np.random.rand(self.size_pop, self.n_dim) - 0.5) * 4
        self.Y = np.array([self.func(x) for x in self.X])

        best_idx = self.Y.argmin()
        self.best_x, self.best_y = self.X[best_idx, :], self.Y[best_idx]
        self.best_X, self.best_Y = self.best_x, self.best_y  # will be deprecated, use lowercase

    def move_to_target(self, idx_individual, x_target):
        '''
        move to target
        called by prey(), swarm(), follow()

        :param idx_individual:
        :param x_target:
        :return:
        '''
        x = self.X[idx_individual, :]
        x_new = x + self.step * np.random.rand() * (x_target - x)
        # x_new = x_target
        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if self.Y[idx_individual] < self.best_Y:
            self.best_x = self.X[idx_individual, :].copy()
            self.best_y = self.Y[idx_individual].copy()

    def move(self, idx_individual):
        '''
        randomly move to a point

        :param idx_individual:
        :return:
        '''
        r = 2 * np.random.rand(self.n_dim) - 1
        x_new = self.X[idx_individual, :] + self.visual * r
        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if self.Y[idx_individual] < self.best_Y:
            self.best_X = self.X[idx_individual, :].copy()
            self.best_Y = self.Y[idx_individual].copy()

    def prey(self, idx_individual):
        '''
        prey
        :param idx_individual:
        :return:
        '''
        for try_num in range(self.max_try_num):
            r = 2 * np.random.rand(self.n_dim) - 1
            x_target = self.X[idx_individual, :] + self.visual * r
            if self.func(x_target) < self.Y[idx_individual]:  # 捕食成功
                self.move_to_target(idx_individual, x_target)
                return None
        # 捕食 max_try_num 次后仍不成功，就调用 move 算子
        self.move(idx_individual)

    def find_individual_in_vision(self, idx_individual):
        # 找出 idx_individual 这条鱼视线范围内的所有鱼
        distances = spatial.distance.cdist(self.X[[idx_individual], :], self.X, metric='euclidean').reshape(-1)
        idx_individual_in_vision = np.argwhere((distances > 0) & (distances < self.visual))[:, 0]
        return idx_individual_in_vision

    def swarm(self, idx_individual):
        # 聚群行为
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            center_individual_in_vision = individual_in_vision.mean(axis=0)
            center_y_in_vision = self.func(center_individual_in_vision)
            if center_y_in_vision * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, center_individual_in_vision)
                return None
        self.prey(idx_individual)

    def follow(self, idx_individual):
        # 追尾行为
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            y_in_vision = np.array([self.func(x) for x in individual_in_vision])
            idx_target = y_in_vision.argmin()
            x_target = individual_in_vision[idx_target]
            y_target = y_in_vision[idx_target]
            if y_target * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, x_target)
                return None
        self.prey(idx_individual)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for epoch in range(self.max_iter):
            for idx_individual in range(self.size_pop):
                self.swarm(idx_individual)
                self.follow(idx_individual)
            self.visual *= self.q
        self.best_X, self.best_Y = self.best_x, self.best_y  # will be deprecated, use lowercase
        return self.best_x, self.best_y




def fitness_func(ga_instance, solution, solution_idx):
    z = solution[0] + 1j * solution[1]
    return 1 / (1 + abs(z ** 6 - 1))


def func(x):
    x1, x2 = x
    z = x1 + 1j * x2
    return -1 / (1 + abs(z ** 6 - 1))

"""
afsa = AFSA(func, n_dim=2, size_pop=50, max_iter=300,
            max_try_num=100, step=0.3, visual=0.3,
            q=0.98, delta=0.5)
# max_try_num - Максимальное количество попыток хищничества
# step - Максимальный коэффициент смещения каждой ступени
# visual - Максимальная дальность восприятия рыбы
# q -Коэффициент затухания дальности восприятия рыбы
# delta -Порог скученности, чем больше, тем легче группировать и замыкать

best_x, best_y = afsa.run()
print(best_x, best_y)


fitness_function = fitness_func

num_generations = 200
num_parents_mating = 4

sol_per_pop = 8
num_genes = 2 #len(function_inputs)

init_range_low = -2
init_range_high = 2

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

"""
all_solutions = {}
start = time.time()
for i in range (100):

    afsa = AFSA(func, n_dim=2, size_pop=100, max_iter=100,
                max_try_num=100, step=0.3, visual=0.3,
                q=0.98, delta=0.5)
    # max_try_num - Максимальное количество попыток хищничества
    # step - Максимальный коэффициент смещения каждой ступени
    # visual - Максимальная дальность восприятия рыбы
    # q -Коэффициент затухания дальности восприятия рыбы
    # delta -Порог скученности, чем больше, тем легче группировать и замыкать

    solution, solution_fitness = afsa.run()


    #solution, solution_fitness, solution_idx = ga_instance.best_solution()
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
            if solution_fitness < all_solutions[pair]:
                all_solutions.pop(pair)
                all_solutions[solution[0], solution[1]] = solution_fitness
            break
    if break_flag == 1:
        continue

    all_solutions[solution[0], solution[1]] = solution_fitness
error = 0

for pair in all_solutions:
    print(pair, all_solutions[pair])
    error = error + (1 + all_solutions[pair]) ** 2

end = time.time() - start
print(end)
print("MSI = ", error/6)

# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# prediction = F(solution)  #np.sum(np.array(function_inputs)*solution)
# print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

# fmin(F_inv, np.array([-2, -2]))

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
