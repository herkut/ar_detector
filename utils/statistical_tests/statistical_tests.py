import numpy as np
import json
from scipy import stats


def cochrans_q_test():
    pass


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


def compare_models_wrt_5x2cv_paired_t_test(results_model1, results_model2, metric='f1'):
    s_sq = np.zeros(5)
    p_mat = np.zeros((5, 2))
    # calculating variance estimates

    for i in range(0, 5):
        p_mat[i][0] = results_model1[i][0][metric] - results_model2[i][0][metric]
        p_mat[i][1] = results_model1[i][1][metric] - results_model2[i][1][metric]
        s_sq[i] = (np.square(p_mat[i][0] - np.mean(p_mat[i])) + np.square(p_mat[i][1] - np.mean(p_mat[i])))

    t = p_mat[0][0] / (np.sqrt(np.sum(s_sq) / 5) if np.sqrt(np.sum(s_sq) / 5)>0 else 1)
    print("t estimation: " + str(t) + ' and t value with 95 confidence interval: ' + str(stats.t.ppf(1 - 0.025, 5)))

    if t > stats.t.ppf(1 - 0.025, 5) or t < -stats.t.ppf(1 - 0.025, 5):
        if np.mean(p_mat) < 0:
            # model 2 is better
            return t, -1
        elif np.mean(p_mat) > 0:
            # model 1 is better
            return t, 1
        else:
            print('Boss, we have an issue')
    else:
        # print('Models are not significantly different')
        return t, 0


def compare_models_wrt_5x2cv_paired_f_test(results_model1, results_model2, metric='f1'):
    s_sq = np.zeros(5)
    p_mat = np.zeros((5, 2))
    # calculating variance estimates

    # metrics = ['sensitivity/recall', 'f1', 'accuracy']

    for i in range(0, 5):
        p_mat[i][0] = results_model1[i][0][metric] - results_model2[i][0][metric]
        p_mat[i][1] = results_model1[i][1][metric] - results_model2[i][1][metric]
        s_sq[i] = (np.square(p_mat[i][0] - np.mean(p_mat[i])) + np.square(p_mat[i][1] - np.mean(p_mat[i])))

    # Prevent divisor to be 0
    f = np.sum(np.square(p_mat)) / ((2 * np.sum(s_sq)) if np.sum(s_sq)>0 else 1)
    print("f estimation: " + str(f) + ' and f value with 95 confidence interval: ' + str(stats.f.ppf(1-0.05, 10, 5)))

    if f < stats.f.ppf(1-0.05, 10, 5):
        # print('Models are not significantly different')
        return f, 0
    else:
        # print('Models are significantly different')
        if np.mean(p_mat) < 0:
            # model 2 is better
            return f, -1
        elif np.mean(p_mat) >= 0:
            # model 1 is better
            return f, 1
        else:
            print('Boss, we have an issue')


def choose_best_models(result_json_model1, result_json_model2):
    model1 = json.load(result_json_model1)
    model2 = json.load(result_json_model2)
    compare_models_wrt_5x2cv_paired_f_test(model1['results'], model2['results'])


if __name__ == '__main__':
    # compare_models_wrt_5x2cv_paired_f_test([],[])
    # print(stats.f.ppf(1-0.05, 10, 5))
    target_drugs = ['Isoniazid', 'Rifampicin', 'Ethambutol', 'Pyrazinamide', 'Streptomycin', 'Ofloxacin', 'Amikacin',
                    'Ciprofloxacin', 'Moxifloxacin', 'Capreomycin', 'Kanamycin']

    results_5x2cv_paired_f_test = '/run/media/herkut/herkut/TB_genomes/ar_detector_results/5x2cv_f_tests/'

    data_representations = ['binary', 'tfrf', 'bm25tfrf']
    models = ['svm_linear', 'svm_rbf', 'lr', 'rf', 'dnn1', 'dnn2']

    for i in range(0, len(target_drugs)):
        print('For ' + target_drugs[i] + ': ')
        for m1 in models:
            m1 = m1 + '_' + data_representations[0]
            for m2 in models:
                m2 = m2 + '_' + data_representations[0]
                if m1 != m2:
                    with open(results_5x2cv_paired_f_test + target_drugs[i] + '/' + m1 + '.json') as json_data:
                        m1_results = json.load(json_data)

                    with open(results_5x2cv_paired_f_test + target_drugs[i] + '/' + m2 + '.json') as json_data:
                        m2_results = json.load(json_data)

                    t_f, res = compare_models_wrt_5x2cv_paired_t_test(m1_results, m2_results, metric='f1')
                    if res == 0:
                        print('Not significantly different: ' + m1 + ' and ' + m2)
                    elif res == 1:
                        print(m1 + ' is better than ' + m2)
                    elif res == -1:
                        print(m2 + ' is better than ' + m1)
