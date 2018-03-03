import argparse
import numpy as np
import matplotlib.pyplot as plt


# image preprocess for part 5

def running_mean(x, N):
    x = np.array(x)
    # insert: Insert values 0 along the given axis before the given indices 0.
    cumsum = np.cumsum(np.insert(x, 0, 0))
    # print("cumsum[N:]",cumsum[N:])
    # print("cumsum[:-N]",cumsum[:-N])
    result = (cumsum[N:] - cumsum[:-N]) / N
    # print("x",x.shape, "cumsum", result.shape)
    return result


def plot_running_mean(rewards_list, filename):
    smoothed_rews = running_mean(rewards_list, 50)

    eps = np.arange(len(rewards_list))
    # print("smoothed_rews",smoothed_rews)
    # print("eps[-len(smoothed_rews):]",eps[-len(smoothed_rews):])
    # G_t
    skipNums = len(smoothed_rews)
    # print("skipNums", skipNums, "eps", eps.shape)
    plt.plot(eps[-skipNums:], smoothed_rews)

    # plt.plot(smoothed_rews)
    # moving average of G_t
    plt.plot(eps, rewards_list, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('../result/' + filename + '.jpg')
