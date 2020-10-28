import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve, precision_score, recall_score


def plot_roc(model, x, y, label=''):
    base = [0 for _ in range(len(y))]
    try:
        if type == 'sklearn':
            res = model.predict_proba(x)[:,1]
        else: # tf or keras
            res = model.predict(x)
    except:
        ValueError('model does not provide predict probability')

    base_fpr, base_tpr, _ = roc_curve(y, base)
    fpr, tpr, _ = roc_curve(y, res)

    plt.plot(base_fpr, base_tpr, linestyle='--', label='base')
    plt.plot(fpr, tpr, marker='.', label=label)

    plt.legend()
    plt.title('roc-curve')

    plt.show()


def plot_pr(model, x, y, type, label=''):
    base = len(y[y==1]) / len(y)
    try:
        if type == 'sklearn':
            res = model.predict_proba(x)[:,1]
        else: # tf or keras
            res = model.predict(x)
    except:
        ValueError('model does not provide predict probability')

    recall, precision, _ = precision_recall_curve(y, res)

    plt.plot([0, 1], [base, base], linestyle='--', label='base')

    plt.plot(recall, precision, marker='.', label=label)
    plt.legend()
    plt.title('pr-curve')
    plt.show()