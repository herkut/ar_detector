import numpy as np
import json
from scipy import stats


def compare_models_wrt_kfold_cross_validated_paired_t_test(arr1, arr2):
    k = len(arr1)

    x = np.zeros(k)

    for i in range(0, k):
        x[i] = arr1[i] - arr2[i]

    m = np.mean(x)
    s = np.subtract(x, m)
    s = np.square(s)
    s = np.sum(s) / (k - 1)
    t = np.sqrt(k) * m / s
    if t > stats.t.ppf(1 - 0.025, k - 1):
        return 1
    elif t < -stats.t.ppf(1 - 0.025, k - 1):
        return -1
    else:
        # print('We found accuracies that are not significantly different')
        return 0


def choose_best_hyperparameters(json_containing_cross_validation_results):
    d = json.load(json_containing_cross_validation_results)
    best_hyperparameter_id = 0
    for i in range(1, len(d['results'])):
        res = compare_models_wrt_kfold_cross_validated_paired_t_test(
            ['results'][best_hyperparameter_id]['validation_accuracies'],
            d['results'][i]['validation_accuracies'])
        if res == -1:
            best_hyperparameter_id = i

    print('Best hyperparameter index: ' + str(best_hyperparameter_id))
    print(d['parameters'][best_hyperparameter_id])

    return best_hyperparameter_id


def compare_models_wrt_5x2cv_paired_t_test(results_model1, results_model2):
    s_sq = np.zeros(5)
    p_mat = np.zeros((5, 2))
    # calculating variance estimates

    for i in range(0, 5):
        p_mat[i][0] = results_model1[i]['test_accuracies'][0] - results_model2[i]['test_accuracies'][0]
        p_mat[i][1] = results_model1[i]['test_accuracies'][1] - results_model2[i]['test_accuracies'][1]
        s_sq[i] = (np.square(p_mat[i][0] - np.mean(p_mat[i])) + np.square(p_mat[i][1] - np.mean(p_mat[i])))

    t = p_mat[1][1] / (np.sum(s_sq) / 5)

    if t > stats.t.ppf(1 - 0.025, 5) or t < -stats.t.ppf(1 - 0.025, 5):
        if np.mean(p_mat) < 0:
            # model 2 is better
            return -1
        elif np.mean(p_mat) > 0:
            # model 1 is better
            return 1
        else:
            print('Boss, we have an issue')
    else:
        # print('Models are not significantly different')
        return 0


def compare_models_wrt_5x2cv_paired_f_test(results_model1, results_model2):
    s_sq = np.zeros(5)
    p_mat = np.zeros((5, 2))
    # calculating variance estimates

    for i in range(0, 5):
        p_mat[i][0] = results_model1[i]['test_accuracies'][0] - results_model2[i]['test_accuracies'][0]
        p_mat[i][1] = results_model1[i]['test_accuracies'][1] - results_model2[i]['test_accuracies'][1]
        s_sq[i] = (np.square(p_mat[i][0] - np.mean(p_mat[i])) + np.square(p_mat[i][1] - np.mean(p_mat[i])))

    f = np.sum(np.square(p_mat)) / (2 * np.sum(s_sq))
    print("f estimation: " + str(f) + ' and f value with 95 confidence interval: ' + str(stats.f.ppf(1-0.05, 10, 5)))
    if f < stats.f.ppf(1-0.05, 10, 5):
        print('Models are not significantly different')
        return 0
    else:
        print('Models are significantly different')
        if np.mean(p_mat) < 0:
            # model 2 is better
            return -1
        elif np.mean(p_mat) > 0:
            # model 1 is better
            return 1
        else:
            print('Boss, we have an issue')


def choose_best_models(result_json_model1, result_json_model2):
    model1 = json.load(result_json_model1)
    model2 = json.load(result_json_model2)
    compare_models_wrt_5x2cv_paired_f_test(model1['results'], model2['results'])


if __name__ == '__main__':
    # compare_models_wrt_5x2cv_paired_f_test([],[])
    print(stats.f.ppf(1-0.05, 10, 5))
