from hyperopt import tpe, rand, anneal, atpe, Trials, hp, fmin, STATUS_OK

from utils import minisom


def som_objective(space):
    # print("Hyper-parameters optimization in function som_fn")
    sig = space['sigma']
    learning_rate = space['learning_rate']
    x = int(space['x'])
    data = space['data']
    val = minisom.MiniSom(x=x,
                          y=x,
                          input_len=data.shape[1],
                          sigma=sig,
                          learning_rate=learning_rate,
                          ).quantization_error(data[0:100, :])
    # print(space)
    # print("Current value {}".format(val))
    return {'loss': val, 'status': STATUS_OK}


def hyper_opt_base(X_train, algo='tpe', verbose=False, show_progressbar=False, max_evals=500):
    print("Hyper-parameters optimization process. The algorithm used is {}.".format(algo))
    if algo.__eq__('tpe'):
        algo = tpe.suggest
    elif algo.__eq__('rand'):
        algo = rand.suggest
    elif algo.__eq__('atpe'):
        algo = atpe.suggest
    elif algo.__eq__('anneal'):
        algo = anneal.suggest
    else:
        print("Default algorithm is tpe")
        algo = tpe.suggest
    space = {
        'sigma': hp.uniform('sigma', 5, 10),
        'learning_rate': hp.uniform('learning_rate', 0.05, 5),
        'x': hp.uniform('x', 20, 50),
        'data': X_train
    }
    trials = Trials()
    # max_evals can be set to 1200, but for speed, we set to 100
    best = fmin(fn=som_objective,
                space=space,
                algo=algo,
                max_evals=max_evals,
                trials=trials,
                verbose=verbose,
                show_progressbar=show_progressbar)
    print('Best: {}'.format(best))


def bayes_opt_base():
    pass