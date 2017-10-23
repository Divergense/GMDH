# coding: utf-8

import numpy as np
import itertools as it


def learn(data_x, data_y, c, F=6, ref_f='lin', crit_f='ssq', max_lvl=3, regularization=False, prnt=True):
    """
    estimation polynomial coefficients of the base function
    max_lvl - layers maximum number
    """
    ref_functions = {'lin': lin, 'mul': mul, 'squ': squ}
    crit_functions = {'ssq': crit_ssq, 'rss': crit_rss, 'prc': crit_prc, 'mimax': crit_mimax, 'nrss': crit_nrss}

    ref_f = ref_functions[ref_f]
    crit_f = crit_functions[crit_f]

    train_y = data_y[c == 1]
    test_y = data_y[c == 0]

    # ----------------------------------------------------------------------------------------------------------------
    n = 0
    #  err_old = np.inf
    x, w_best, err_best, ij, n = step_learn(data_x, train_y, test_y, c, n, ref_f, crit_f, F, prnt)
    m = np.hstack([ij, w_best[:F], err_best[:F][:, np.newaxis]])[np.newaxis]
    while n < max_lvl:                                                  # stopping criterion
        #  err_old = err_best.mean()
        if regularization:
            x = np.hstack([x, data_x])
        x, w_best, err_best, ij, n = step_learn(x, train_y, test_y, c, n, ref_f, crit_f, F, prnt)
        m2 = np.hstack([ij, w_best[:F], err_best[:F][:, np.newaxis]])
        m = np.vstack([m, [m2]])
    return m


def step_learn(x, train_y, test_y, c, n, ref_f, crit_f, F, prnt):       # calculating average error of the F best models
    n += 1
    x_new = []
    # err_old = err_best.mean()
    w_all, errors, ij = find_w(x, train_y, test_y, c, ref_f, crit_f)    # list of matrixes of all features
    w_best, err_best, ij = selection(w_all, errors, ij, F)              # ij is changed, select the bests
    if prnt:
        print('Step {:4d} \tAVE = {:6.5f} \tmin = {:6.5f}'.format(n, err_best.mean(), err_best[0]))
    for k, i in enumerate(ij):
        x_new.append(value_func(ref_f(x[:, i]), w_best[k]))
    x = np.array(x_new).T
    return x, w_best, err_best, ij, n


def find_w(x, train_y, test_y, c, ref_f, crit_f):
    """
    x - data matrix
    """
    ij = np.array(list(it.combinations(range(x.shape[1]), 2)), dtype=np.int)  # индексы всех возможных пар признаков
    w = []  # список весов опорных функций
    errors = []
    for i in ij:
        x_ij = ref_f(x[:, i])  # зависит от опорной функции
        w_ij = np.dot(np.linalg.inv(np.dot(x_ij[c == 1].T, x_ij[c == 1])), np.dot(x_ij[c == 1].T, train_y))
        # w_ij = np.linalg.lstsq(x_ij[c == 1], train_y)[0]
        # np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
        error = crit_f(value_func(x_ij[c == 0], w_ij), test_y)  # c - global?
        w.append(w_ij)
        errors.append(error)  # список ошибок каждой модели
    return np.array(w), np.array(errors), ij


def crit_ssq(y_p, y_tst):
    return (np.sum((y_p - y_tst) ** 2) / len(y_tst)) ** 0.5


def crit_rss(y_p, y_tst):
    return np.sum((y_p - y_tst) ** 2)


def crit_prc(y_p, y_tst):
    return np.sum(100 * np.abs(y_tst - y_p) / y_tst) / len(y_tst)


def crit_mimax(y_p, y_tst):
    return np.max(np.abs(y_tst - y_p))


def crit_nrss(y_p, y_tst):
    return np.sum((y_p - y_tst) ** 2) / np.sum(y_tst ** 2)


def selection(w, errors, ij, F):
    """
    F - количество лучших моделей
    """
    ind = np.argsort(errors)  # [:n]
    best_err = errors[ind]
    best_w = w[ind]
    ij = ij[ind][:F]
    return best_w, best_err, ij  # deleted [:F]


def value_func(x, w):
    return np.dot(x, w)  # w - вектор весов, x - матрица из единиц и 2-х переменных


def lin(x):
    return np.vstack([np.ones(x.shape[0]), x[:, 0], x[:, 1]]).T


def mul(x):
    return np.vstack([np.ones(x.shape[0]), x[:, 0], x[:, 1], x[:, 0] * x[:, 1]]).T


def squ(x):
    return np.vstack([np.ones(x.shape[0]), x[:, 0], x[:, 1], x[:, 0] * x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2]).T


def predict_reg(res_matrix, point0, ref_f='lin', lvl=1, num=0):
    """для моделей с регуляризацией"""
    ref_functions = {'lin': (lin, 3), 'mul': (mul, 4), 'squ': (squ, 6)}
    ref_f, num_ft = ref_functions[ref_f]

    nf = res_matrix.shape[1]

    ij = [np.array(res_matrix[lvl][num:num + 1, :2].ravel(), dtype=np.int)[np.newaxis, :]]  # первый элемент матрица!
    ii = ij[0]

    w_ij = [res_matrix[lvl][num:num + 1, 2:2 + num_ft]]
    for i in range(lvl - 1, -1, -1):
        # ii проверять на превышение количества
        ii = np.array(res_matrix[i][ii[ii < nf], :2], dtype=np.int)
        ij.append(ii)
        i3 = np.array(map(lambda x: (x == res_matrix[i][:, :2]).all(1), ii))
        w_ij.append(np.array(map(lambda ind: res_matrix[i][ind, 2:2 + num_ft].ravel(), i3)))

    if len(point0.shape) == 1:
        point0 = point0[np.newaxis, :]

    point = map(lambda ind, w: value_func(ref_f(point0[:, ind]), w), ij[-1], w_ij[-1])  # ind!!!
    point = np.array(point).T

    for i in range(lvl - 1, -1, -1):
        j = 0
        point2 = []
        for ind in ij[i]:
            if (ind >= nf).any():  # на основании, что превышающий признак только 1 и всегда под индексом 1
                point2.append(np.vstack([point[:, j], point0[:, ind[1] - nf]]).T)
                j += 1
            else:
                point2.append(point[:, j:j + 2])
                j += 2

        point = map(lambda p, w: value_func(ref_f(p), w), point2, w_ij[i])
        point = np.array(point).T
    return point.ravel()


def predict(res_matrix, point, ref_f='lin', lvl=1, num=0):
    """для моделей без регуляризации"""
    ref_functions = {'lin': (lin, 3), 'mul': (mul, 4), 'squ': (squ, 6)}
    ref_f, num_ft = ref_functions[ref_f]

    ij = [np.array(res_matrix[lvl][num:num + 1, :2].ravel(), dtype=np.int)[np.newaxis, :]]  # первый элемент матрица
    ii = ij[0]

    w_ij = [res_matrix[lvl][num:num + 1, 2:2 + num_ft]]
    for i in range(lvl - 1, -1, -1):
        ii = np.array(res_matrix[i][ii.ravel(), :2], dtype=np.int)  # лучше сразу w_ij?
        ij.append(ii)
        i3 = np.array(map(lambda x: (x == res_matrix[i][:, :2]).all(1), ii))
        w_ij.append(np.array(map(lambda ind: res_matrix[i][ind, 2:2 + num_ft].ravel(), i3)))

    if len(point.shape) == 1:
        point = point[np.newaxis, :]

    point = map(lambda ind, w: value_func(ref_f(point[:, ind]), w), ij[-1], w_ij[-1])  # список(!) вектор-признаков
    point = np.array(point).T  # матрица вектор-признаков (уже столбцы)

    for i in range(lvl - 1, -1, -1):
        point = [point[:, j:j + 2] for j in np.arange(0, point.shape[1], 2)]
        point = map(lambda p, w: value_func(ref_f(p), w), point, w_ij[i])
        point = np.array(point).T
    return point.ravel()


def gabor(x, m):
    """
    m - степень полинома
    C^k_{k+n-1} - число переменных на k порядке полинома
    kl←{+/⍵!⍺+⍵-1}
    """
    s = map(lambda n: np.array(list(it.combinations_with_replacement(range(x.shape[1]), n))),
            np.arange(2, m + 1))
    x2 = x.T
    for i in s:
        for j in i:
            x2 = np.vstack([x2, np.multiply.reduce(x[:, j], 1)])
    return x2.T


def normalization(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)
