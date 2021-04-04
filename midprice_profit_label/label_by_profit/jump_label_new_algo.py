# use python3


import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def time2int(time_str: str) -> int:
    """Transform '01:57:00' to (int)157"""
    return int(time_str[:2] + time_str[3:5])


def time2str(time_int: int) -> str:
    """Transform 157 to '01:57:00'"""
    padded_str = str(time_int).zfill(4)  # 157 becomes "0157"
    return padded_str[:2] + ":" + padded_str[2:4] + ":00"


def narrow_adjust(closing_prices, leftmost_min_index, leftmost_max_index, curr_min, curr_max, window_lborder,
                  window_rborder):
    best_min_index = leftmost_min_index
    best_max_index = leftmost_max_index
    if leftmost_min_index < leftmost_max_index:
        while (closing_prices[best_min_index + 1] == curr_min):
            best_min_index += 1
        while (closing_prices[best_max_index - 1] == curr_max):
            best_max_index -= 1
    else:
        while (closing_prices[best_min_index - 1] == curr_min):
            best_min_index -= 1
        while (closing_prices[best_max_index + 1] == curr_max):
            best_max_index += 1
    return best_min_index, best_max_index


def plot_graph(single_date,
               closing_prices,
               min_max_pairs,
               min_close_price,
               max_close_price,
               hyperparams,
               dark_mode=False):
    if dark_mode:
        plt.style.use('dark_background')
    figure(figsize=(48, 10), dpi=100)

    ax = plt.subplot(1, 1, 1)
    for pair in min_max_pairs:
        # print(pair)
        # Green when price surges, red when price drops
        if dark_mode:
            curr_color = (.2, .45, .2) if pair[0] < pair[1] else (.4, .2, .2)
        else:
            curr_color = (.7, 1, .7) if pair[0] < pair[1] else (1, .7, .7)
        ax.add_patch(
            patches.Rectangle((min(pair[0], pair[1]), min_close_price),
                              abs(pair[0] - pair[1]),
                              max_close_price - min_close_price + 3,
                              color=curr_color))

    if dark_mode:
        plt.plot(closing_prices, color="#99ccff")
    else:
        plt.plot(closing_prices)
    plt.legend(['Closing price'], fontsize=20)
    plt.title(f'New Algorithm ({single_date.strftime("%Y-%m-%d")})\n' +
              f'No. of green/red stripes: {len(min_max_pairs)}, ' + f'Window size: {hyperparams[0]}, ' +
              f'Slope threshold: {hyperparams[1]}, ' + f'Jump size threshold: {hyperparams[2]}',
              fontsize=30)
    plt.xlabel('Minutes since 00:00:00', fontsize=25)
    plt.xticks(fontsize=18)
    plt.ylabel('Closing price', fontsize=25)
    plt.yticks(fontsize=18)
    plt.savefig("figures_new_algo/" + single_date.strftime("%Y-%m-%d") +
                f'_{hyperparams[0]}__{hyperparams[1]}__{hyperparams[2]}_' + ('_(dark)' if dark_mode else '_(light)') +
                '.png')
    plt.clf()


def main(window_size, slope_threshold, jump_size_threshold):
    # window_size = 5  # hyperparameter window size
    # slope_threshold = 0.1  # hyperparameter slope threshold
    # jump_size_threshold = 1.0  # hyperparameter jump size threshold
    hyperparams = (window_size, slope_threshold, jump_size_threshold)

    start_date = date(2010, 3, 24)
    end_date = date(2010, 3, 27)
    for single_date in date_range(start_date, end_date):
        df = pd.read_csv(single_date.strftime("%Y-%m-%d") + '.csv')
        df.sort_values(by='dt')  # don't need?

        times = df['tm'].values.tolist()  # the time (hr:min:sec) column
        closing_prices = df['close'].values.tolist()  # the closing price column
        max_close_price = max(closing_prices)
        min_close_price = min(closing_prices)

        start_time: int = time2int(times[0])
        end_time: int = time2int(times[-1])
        window_lborder: int = start_time
        window_rborder: int = start_time + window_size  # upperbound to be excluded
        min_max_pairs = []  # list of start-end index pairs whose area between should be colored red/green

        while window_lborder < end_time:
            window_rborder = min(window_rborder, end_time)
            curr_slice = closing_prices[window_lborder:window_rborder]
            if len(curr_slice) == 0:
                break
            curr_min: float = min(curr_slice)
            curr_max: float = max(curr_slice)
            if curr_min == curr_max:
                window_lborder = window_rborder
                window_rborder += window_size
                continue

            leftmost_min_index: int = closing_prices.index(curr_min, window_lborder, window_rborder)
            leftmost_max_index: int = closing_prices.index(curr_max, window_lborder, window_rborder)
            best_min_index, best_max_index = narrow_adjust(closing_prices, leftmost_min_index, leftmost_max_index,
                                                           curr_min, curr_max, window_lborder, window_rborder)

            if ((curr_max - curr_min) / abs(best_min_index - best_max_index) > slope_threshold) and (
                (curr_max - curr_min) >= jump_size_threshold):
                min_max_pairs.append([best_min_index, best_max_index])
                window_lborder = max(best_min_index, best_max_index)
                window_rborder = window_lborder + window_size
            else:
                window_lborder = window_rborder
                window_rborder += window_size

        plot_graph(single_date,
                   closing_prices,
                   min_max_pairs,
                   min_close_price,
                   max_close_price,
                   hyperparams,
                   dark_mode=True)


if __name__ == '__main__':
    count = 0
    for i in range(1, 16):  # slope
        for j in range(8, 26):  # jump size
            main(5, i / 10, j / 10)
            count += 1
            print(f">>>>>>{count*100/(15*18):.2f}% Done...\n")
