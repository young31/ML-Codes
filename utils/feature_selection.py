import numpy as np

# TODO
## change predict / predict_proba depending on problem
def fitness(tr_X, tr_y, val_X, val_y, model, criterion):
    def _fitness(enc):
        x = tr_X[extract(enc, data=tr_X)]
        model.fit(x, tr_y)
        return criterion(val_y, model.predict_proba(val_X[extract(enc, data=tr_X)])[:,1])
    return _fitness

def crossover(xs, n=None):
    if n is None:
        n = len(xs)
    new = None
    for i in range(n):
        a, b = np.random.choice(range(len(xs)), 2, replace=False)
        x1 = xs[a]; x2 = xs[b]
        point = np.random.choice(range(len(x1)))
        newx1 = np.hstack([x1[:point], x2[point:]])
        newx2 = np.hstack([x2[:point], x1[point:]])
        if new is None:
            new = np.vstack([newx1, newx2])
        else:
            new = np.vstack([new, newx1, newx2])
    return new

def mutate(xs, n=None):
    if n is None:
        n = int(np.sqrt(len(xs))) 
    new = None
    for i in range(len(xs)):
        points = np.random.choice(range(xs.shape[1]), n, replace=False)
        newx = xs[i].copy()
        for point in points:
            newx[point] = int(np.logical_not(newx[point]))
            
        if new is None:
            new = newx
        else:
            new = np.vstack([new, newx])
    return new

def extract(enc, data):
    return data.columns[enc==1]

def encode(features, data):
    res = np.zeros(data.shape[1])
    for i, c in enumerate(data.columns):
        if c in features:
            res[i] = 1
    return res


def GA_select(tr_X, tr_y, val_X, val_y, model, criterion,
             n_base=10, n_iter=10):
    '''
    support padnas dataframe not numpy array
    now elite selection only, rhs or tournament will be contained
    '''
    n_base = n_base
    n_iter = n_iter

    bases = np.random.randint(0, 2, (n_base, tr_X.shape[1]))
    alls = np.ones(tr_X.shape[1])

    bases = np.vstack([bases, alls])

    final_score = 0

    for _ in range(n_iter):
        res = {}
        cross_base = crossover(bases)
        mut_base = mutate(bases)

        scores1 = list(map(fitness(tr_X, tr_y, val_X, val_y, model=model, criterion=criterion), bases))
        scores2 = list(map(fitness(tr_X, tr_y, val_X, val_y, model=model, criterion=criterion), cross_base))
        scores3 = list(map(fitness(tr_X, tr_y, val_X, val_y, model=model, criterion=criterion), mut_base))

        scores = scores1+scores2+scores3
        bases = np.vstack([bases, cross_base, mut_base])

        for i in range(len(bases)):
            res[scores[i]] = bases[i]

        gen = []
        for s in sorted(res, reverse=True)[:n_base]:
            gen.append(res[s])

        bases = np.array(gen)
        best_score = sorted(res, reverse=True)[0]
        print(f'iter: {_+1}', best_score)

        if best_score > final_score:
            final_score = best_score
            features = extract(res[final_score], tr_X)
        else:
            break
            
    return features, final_score
    