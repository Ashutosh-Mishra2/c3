from scipy.optimize import minimize as minimize
import cma.evolution_strategy as cma
# from nevergrad.optimization import registry as algo_registry


def lbfgs(x0, goal_fun, grad_fun):
    # TODO print from the log not from hear
    options = {'disp': True}
    minimize(
        goal_fun,
        x0,
        jac=grad_fun,
        method='L-BFGS-B',
        options=options
    )


def cmaes(x0, goal_fun):
    settings = {}
    es = cma.CMAEvolutionStrategy(x0, 0.1, settings)
    iter = 0
    while not es.stop():
        samples = es.ask()
        if iter == 0:
            samples.append(x0)
        solutions = []
        for sample in samples:
            goal = goal_fun(sample)
            solutions.append(goal)
        es.tell(samples, solutions)
        es.disp()
        iter += 1


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
