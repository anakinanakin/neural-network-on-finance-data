import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def read_table(filename):
    df = pd.read_csv(filename)    
    return df

def result_allfactor_effect(df, result, order=False, factor_filter=None):
    '''
    one result (e.g. f1 or acc)
    for all experiment factor
    plot the curve diagram for rank
    '''
    plt.figure(figsize=(16,16), dpi=80)

    df = df.sort_values(result,ascending=order)
    if factor_filter:
        df = df[factor_filter]

    plt.cla()
    plt.clf()
    cols = df.columns
    plt.suptitle(result)

    # loop for all column in factor_filter
    for i, col in enumerate(cols):
        plt.subplot(len(cols), 1, i+1)
        params = pd.unique(df[col])
        params_curve = { p:[0] for p in params}
        # loop the ranking for all value in each column
        for val in df[col]:
            for k in params_curve: params_curve[k].append(params_curve[k][-1])
            params_curve[val][-1] = params_curve[val][-1]+1
        plt.title(col)
        for k in sorted(params_curve.keys()):
            plt.plot(params_curve[k], label=k)
        plt.legend()
    # plt.show()
    plt.savefig(result+'_racing.png')

def results_correlation(df, result1, result2):
    '''
    plot correlation for two result (e.g. f1 and acc)
    scatter plot 
    '''

    plt.figure(figsize=(8,8), dpi=80)
    plt.cla()
    plt.clf()

    x = df[result1]
    y = df[result2]
    plt.xlabel(result1)
    plt.ylabel(result2)
    plt.scatter(x, y)
    # plt.show()
    plt.savefig(result1+'_'+result2+'_correlation.png')

def result_single_factor_dist(df, result, factors):
    '''
    one result (e.g. f1 of acc)
    one experiment factor
    plot scatter factor value and result
    '''

    for f in factors:

        plt.figure(figsize=(8,8), dpi=80)
        plt.cla()
        plt.clf()

        if f == 'label_threshold':
            ax = plt.gca()
            ax.set_xlim([0,0.002])

        x = df[f]
        y = df[result]
        plt.xlabel(f)
        plt.ylabel(result)
        plt.scatter(x, y)
        # plt.show()
        plt.savefig(result+'_'+f+'_single_factor.png')
    


def main():
    df = read_table('evaluation.csv')

    # result_allfactor_effect(df, 'valid_loss_min', order=True, factor_filter=['k','feature_num','label_threshold'])
    # result_allfactor_effect(df, 'valid_acc_max', order=False, factor_filter=['k','feature_num','label_threshold'])
    # result_allfactor_effect(df, 'valid_f1_max', order=False, factor_filter=['k','feature_num','label_threshold'])
    # result_allfactor_effect(df, 'valid_loss_500_epoch', order=True, factor_filter=['k','feature_num','label_threshold'])
    # result_allfactor_effect(df, 'valid_acc_500_epoch', order=False, factor_filter=['k','feature_num','label_threshold'])
    # result_allfactor_effect(df, 'valid_f1_500_epoch', order=False, factor_filter=['k','feature_num','label_threshold'])
    
    # results_correlation(df, 'valid_f1_max', 'valid_acc_max')
    # results_correlation(df, 'valid_loss_min', 'valid_acc_max')
    # results_correlation(df, 'valid_loss_min', 'valid_f1_max')
    # results_correlation(df, 'valid_f1_500_epoch', 'valid_acc_500_epoch')
    # results_correlation(df, 'valid_loss_500_epoch', 'valid_acc_500_epoch')
    # results_correlation(df, 'valid_loss_500_epoch', 'valid_f1_500_epoch')

    result_single_factor_dist(df, 'valid_f1_max', ['k','feature_num','label_threshold'])
    result_single_factor_dist(df, 'valid_loss_min', ['k','feature_num','label_threshold'])
    result_single_factor_dist(df, 'valid_acc_max', ['k','feature_num','label_threshold'])
    result_single_factor_dist(df, 'valid_f1_100_epoch', ['k','feature_num','label_threshold'])
    result_single_factor_dist(df, 'valid_loss_100_epoch', ['k','feature_num','label_threshold'])
    result_single_factor_dist(df, 'valid_acc_100_epoch', ['k','feature_num','label_threshold'])

if __name__ == '__main__':
    main()