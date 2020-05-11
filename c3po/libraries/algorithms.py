from scipy.optimize import minimize as minimize
import cma.evolution_strategy as cma
import numpy as np
# from nevergrad.optimization import registry as algo_registry
import adaptive
import copy

algorithms = dict()
def algo_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    algorithms[str(func.__name__)] = func
    return func


@algo_reg_deco
def single_eval(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    fun(x0)


@algo_reg_deco
def grid2D(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    #TODO generalize grid to take any  number of dymensions

    if 'points' in options:
        points = options['points']
    else:
        points = 100

    # probe_list = []
    # if 'probe_list' in options:
    #     for x in options['probe_list']:
    #         probe_list.append(eval(x))

    # if 'init_point' in options:
    #     init_point = bool(options.pop('init_point'))
    #     if init_point:
    #         probe_list.append(x0)

    bounds = options['bounds'][0]
    bound_min = bounds[0]
    bound_max = bounds[1]
    # probe_list_min = np.min(np.array(probe_list)[:,0])
    # probe_list_max = np.max(np.array(probe_list)[:,0])
    # bound_min = min(bound_min, probe_list_min)
    # bound_max = max(bound_max, probe_list_max)
    xs = np.linspace(bound_min, bound_max, points)

    bounds = options['bounds'][1]
    bound_min = bounds[0]
    bound_max = bounds[1]
    # probe_list_min = np.min(np.array(probe_list)[:,1])
    # probe_list_max = np.max(np.array(probe_list)[:,1])
    # bound_min = min(bound_min, probe_list_min)
    # bound_max = max(bound_max, probe_list_max)
    ys = np.linspace(bound_min, bound_max, points)

    # for p in probe_list:
    #     fun(p)

    for x in xs:
        for y in ys:
            if 'wrapper' in options:
                val = copy.deepcopy(options['wrapper'])
                val[val.index('x')] = x
                val[val.index('y')] = y
                fun([val])
            else:
                fun([x, y])


@algo_reg_deco
def adaptive_scan(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):

    if 'accuracy_goal' in options:
        accuracy_goal = options['accuracy_goal']
    else:
        accuracy_goal = 0.5
    print("accuracy_goal: " + str(accuracy_goal))

    probe_list = []
    if 'probe_list' in options:
        for x in options['probe_list']:
            probe_list.append(eval(x))

    if 'init_point' in options:
        init_point = bool(options.pop('init_point'))
        if init_point:
            probe_list.append(x0)

    # TODO make adaptive scan be able to do multidimensional scan
    bounds = options['bounds'][0]
    bound_min = bounds[0]
    bound_max = bounds[1]
    probe_list_min = min(probe_list)
    probe_list_max = max(probe_list)
    bound_min = min(bound_min, probe_list_min)
    bound_max = max(bound_max, probe_list_max)
    print(" ")
    print("bound_min: " + str((bound_min)/(2e9 * np.pi)))
    print("bound_max: " + str((bound_max)/(2e9 * np.pi)))
    print(" ")
    def fun1d(x): return fun([x])

    learner = adaptive.Learner1D(fun1d, bounds=(bound_min, bound_max))

    if probe_list:
        for x in probe_list:
            print("from probe_list: " + str(x))
            tmp = learner.function(x)
            print("done\n")
            learner.tell(x, tmp)

    runner = adaptive.runner.simple(
        learner,
        goal=lambda learner_: learner_.loss() < accuracy_goal
    )


@algo_reg_deco
def lbfgs(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    # TODO print from the log not from hear
    options.update({'disp': True})
    minimize(
        fun_grad,
        x0,
        jac=grad_lookup,
        method='L-BFGS-B',
        options=options
    )


@algo_reg_deco
def cmaes(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    custom_stop = False
    if 'noise' in options:
        noise = float(options.pop('noise'))
    else:
        noise = 0

    if 'init_point' in options:
        init_point = bool(options.pop('init_point'))
    else:
        init_point = False

    if 'spread' in options:
        spread = float(options.pop('spread'))
    else:
        spread = 0.1

    if 'stop_at_convergence' in options:
        sigma_conv = int(options.pop('stop_at_convergence'))
        sigmas = []
        shrinked_stop = True

    settings = options

    es = cma.CMAEvolutionStrategy(x0, spread, settings)
    iter = 0
    while not es.stop():

        if shrinked_stop:
            sigmas.append(es.sigma)
            if iter > sigma_conv:
                if(
                    all(
                        sigmas[-(i+1)]<sigmas[-(i+2)]
                        for i in range(sigma_conv-1)
                    )
                ):
                    print(
                        f'C3:STATUS:Shrinked cloud for {sigma_conv} steps. '
                        'Switching to gradients.'
                    )
                    break

        samples = es.ask()
        if init_point and iter == 0:
            samples.insert(0,x0)
            print('C3:STATUS:Adding initial point to CMA sample.')
        solutions = []
        for sample in samples:
            goal = fun(sample)
            if noise:
                goal = goal + (np.random.randn() * noise)
            solutions.append(goal)
        es.tell(samples, solutions)
        es.disp()

        iter += 1
    return es.result.xbest


@algo_reg_deco
def cma_pre_lbfgs(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    x1 = cmaes(x0, fun, options=options['cmaes'])
    lbfgs(
        x1, fun_grad=fun_grad, grad_lookup=grad_lookup, options=options['lbfgs']
    )

# def oneplusone(x0, goal_fun):
#     optimizer = algo_registry['OnePlusOne'](instrumentation=x0.shape[0])
#     while True:
#         # TODO make this logging happen elsewhere
#         # self.logfile.write(f"Batch {self.evaluation}\n")
#         # self.logfile.flush()
#         tmp = optimizer.ask()
#         samples = tmp.args
#         solutions = []
#         for sample in samples:
#             goal = goal_fun(sample)
#             solutions.append(goal)
#         optimizer.tell(tmp, solutions)
#
#     # TODO deal with storing best value elsewhere
#     # recommendation = optimizer.provide_recommendation()
#     # return recommendation.args[0]
