import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_metric_df(ll, metric='evaluation_dice_coefficient_vs_iterations'):
    return pd.concat(ll)[metric]

def plot_fold_logs(df, metric='loss', nfolds=5, best_fn=np.min, argbest_fn=np.argmin, test_df=None, show_test=False):
    nfolds = 5
    fig, axs = plt.subplots(nfolds, figsize=(10, 25))
    val_metric = 'val_' + metric
    if show_test and test_df is not None:
        test_dice = get_metric_df(test_df, metric='evaluation_dice_coefficient_vs_iterations').values.flatten()
        test_loss = get_metric_df(test_df, metric='evaluation_loss_vs_iterations').values.flatten()
    for i in range(nfolds):
        axs[i].plot(df[i]['loss'], color='lightsteelblue', label='train loss')
        axs[i].plot(df[i]['val_loss'], color='orange', label='val loss')

        best_val_loss = np.argmin(df[i]['val_loss'])
        best_metric = best_fn(df[i][val_metric])
        argbest_metric = argbest_fn(df[i][val_metric])

        best_dice = np.max(df[i]['val_dice_coefficient'])
        argbest_dice = np.argmax(df[i]['val_dice_coefficient'])
        early_dice = df[i]['val_dice_coefficient'].loc[argbest_metric]

        axs[i].axhline(y=df[i]['val_loss'][argbest_metric], color='goldenrod', linestyle='--', alpha=0.7,
                       label=f'val_loss at early_stop')
        axs[i].plot(df[i]['val_dice_coefficient'], color='seagreen', label='val dice')
        axs[i].axvline(x=argbest_metric, color='red', linestyle='dotted', alpha=0.7, label=f'Early stop val_loss')
        axs[i].axvline(x=argbest_dice, color='green', linestyle='dotted', alpha=0.5, label=f'Early stop dice')

        if show_test:
            axs[i].axhline(y=test_dice[i], color='navy', linestyle='--', label=f'test dice {100 * test_dice[i]:.2f}')
            axs[i].axhline(y=test_loss[i], color='pink', linestyle='--', label=f'test loss {test_loss[i]:.4f}')
        axs[i].axhline(y=best_dice, color='green', linestyle='--', label=f'best val dice {100 * best_dice:.2f}')

        axs[i].set_title(f'Fold {i + 1}')
        axs[i].legend(bbox_to_anchor=(1., 1.0))
        # axs[i, 0].set_title('Axis [0, 0]')

    # for ax in axs.flat:
    #    ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    plt.tight_layout()


def plot_fold_logs(df, metric='loss', nfolds=5, best_fn=np.min, argbest_fn=np.argmin, test_df=None, show_test=False):
    nfolds = 5
    fig, axs = plt.subplots(nfolds, figsize=(10, 25))
    val_metric = 'val_' + metric
    if show_test and test_df is not None:
        test_dice = get_metric_df(test_df, metric='evaluation_dice_coefficient_vs_iterations').values.flatten()
        test_loss = get_metric_df(test_df, metric='evaluation_loss_vs_iterations').values.flatten()
    for i in range(nfolds):
        axs[i].plot(df[i]['loss'], color='lightsteelblue', label='train loss')
        axs[i].plot(df[i]['val_loss'], color='orange', label='val loss')

        best_val_loss = np.argmin(df[i]['val_loss'])
        best_metric = best_fn(df[i][val_metric])
        argbest_metric = argbest_fn(df[i][val_metric])

        best_dice = np.max(df[i]['val_dice_coefficient'])
        argbest_dice = np.argmax(df[i]['val_dice_coefficient'])
        early_dice = df[i]['val_dice_coefficient'].loc[argbest_metric]

        axs[i].axhline(y=df[i]['val_loss'][argbest_metric], color='goldenrod', linestyle='--', alpha=0.7,
                       label=f'val_loss at early_stop')
        axs[i].plot(df[i]['val_dice_coefficient'], color='seagreen', label='val dice')
        axs[i].axvline(x=argbest_metric, color='red', linestyle='dotted', alpha=0.7, label=f'Early stop val_loss')
        axs[i].axvline(x=argbest_dice, color='green', linestyle='dotted', alpha=0.5, label=f'Early stop dice')

        if show_test:
            axs[i].axhline(y=test_dice[i], color='navy', linestyle='--', label=f'test dice {100 * test_dice[i]:.2f}')
            axs[i].axhline(y=test_loss[i], color='pink', linestyle='--', label=f'test loss {test_loss[i]:.4f}')
        axs[i].axhline(y=best_dice, color='green', linestyle='--', label=f'best val dice {100 * best_dice:.2f}')

        axs[i].set_title(f'Fold {i + 1}')
        axs[i].legend(bbox_to_anchor=(1., 1.0))

    plt.tight_layout()