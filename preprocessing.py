import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

stock_list = ['2317']
dates = ['20240328', '20240329', '20240401', '20240402', '20240403']
horizon_k = [10, 20, 50, 100, 200, 300]
alpha = 0.001

# Retaining exclusively the events occurring between 09:30 and 13:00
# Normalization
#   - Calculate mean and var from previous day
#   - Separated normalizer for each feature
#   - Separated normalizer for each stock
# Data labeling

def get_mid_lob(lob_df):
    # from 09:30 to 13:00, timestamp between 3600s to 16200s.
    t_start = 3600 * (10**6)
    t_end = 16200 * (10**6)
    range_lob_df = lob_df.loc[ (lob_df['timestamp'] >= t_start) & (lob_df['timestamp'] <= t_end) ]

    return range_lob_df

def draw_line_chart(series, save_name):
    plt.plot(list(range(len(series))), series)
    plt.savefig(save_name)
    plt.clf()
    plt.cla()

def draw_bar_chart(series, save_name):
    num_fall = len(series[series == -1])
    num_stay = len(series[series == 0])
    num_rise = len(series[series == 1])
    plt.bar(['-1', '0', '1'], [num_fall, num_stay, num_rise], width = 0.4)
    plt.savefig(save_name)
    plt.clf()
    plt.cla()

def draw_segmented_line_chart(df_pt_label, k, save_name):
    import seaborn as sns

    fig, ax1 = plt.subplots(figsize=(12, 4))
    df_pt_label = df_pt_label[:2000]
    
    ax1.plot(list(range(len(df_pt_label))), df_pt_label['mid_price'] + -1*df_pt_label['mid_price'].min(), label="mid_prices", color='blue', linewidth=1)
    print("test1")
    sns.barplot(x=list(range(len(df_pt_label))), y=[(df_pt_label['mid_price'] + -1*df_pt_label['mid_price'].min()).max()] * len(df_pt_label),
            hue='label_'+str(k), alpha=0.5, palette='inferno', dodge=False, data=df_pt_label, ax=ax1).set(xticklabels=[]) #
    print("test2")
    for bar in ax1.patches: # optionally set the bars to fill the complete background, default seaborn sets the width to about 80%
        bar.set_width(1)
    
    plt.legend(bbox_to_anchor=(1.02, 1.05) , loc='upper left')
    plt.tight_layout()
    plt.savefig(save_name)
    print('test3')
    plt.clf()
    plt.cla()
    plt.close()

if __name__ == "__main__":
    Path("preprocessed").mkdir(exist_ok=True)

    # normalization using previous day mean and std
    for s_idx in stock_list:
        for i in range(1,len(dates)): #
            date = dates[i]
            prev_date = dates[i-1]

            lob_df = pd.read_csv("data/%s_%s_.csv" % (date, s_idx))
            prev_lob_df = pd.read_csv("data/%s_%s_.csv" % (prev_date, s_idx))

            cols = lob_df.columns
            range_lob_df = get_mid_lob(lob_df)
            prev_range_lob_df = get_mid_lob(prev_lob_df)

            # remove rows with all zero
            range_lob_df = range_lob_df.loc[ range_lob_df[cols[:-1]].sum(axis=1) > 0 ]
            prev_range_lob_df = prev_range_lob_df.loc[ prev_range_lob_df[cols[:-1]].sum(axis=1) > 0 ]

            # mean and std of previous day
            prev_mean = prev_range_lob_df[cols[:-1]].mean()
            prev_std = prev_range_lob_df[cols[:-1]].std()

            normalized_lob_df = pd.DataFrame(columns=cols[:-1])
            normalized_lob_df[cols[:-1]] = (range_lob_df[cols[:-1]] - prev_mean)/prev_std
            
            # mid price
            temp_df = pd.DataFrame()
            temp_df['mid_price'] = (normalized_lob_df['買進價1'] + normalized_lob_df['賣進價1'])/2
            normalized_lob_df['mid_price'] = temp_df['mid_price']

            # mid price smoothing
            for k in horizon_k:
                temp_df['mm_'+str(k)] = temp_df['mid_price']  # mean of previous k ticks
                temp_df['mp_'+str(k)] = temp_df['mid_price']  # mean of next k ticks 
                for r in range(k):
                    temp_df['mm_'+str(k)] += temp_df['mid_price'].shift(periods=(r+1)).fillna(0)
                    temp_df['mp_'+str(k)] += temp_df['mid_price'].shift(periods=-1*(r+1)).fillna(0)
                
                temp_df['mm_'+str(k)] = temp_df['mm_'+str(k)]/k
                temp_df['mp_'+str(k)] = temp_df['mp_'+str(k)]/k
            temp_df = temp_df[max(horizon_k):-1*max(horizon_k)]
            normalized_lob_df = normalized_lob_df[max(horizon_k):-1*max(horizon_k)]
            # for k in horizon_k:
            #     draw_line_chart(temp_df['mm_'+str(k)], 'mm_%d_%s.png' % (k, date))
            #     draw_line_chart(temp_df['mp_'+str(k)], 'mp_%d_%s.png' % (k, date))

            # labeling
            ranges = [-10**6, -1*alpha, alpha, 10**6 ]
            for k in horizon_k:
                lt_k = (temp_df['mp_'+str(k)] - temp_df['mm_'+str(k)])/temp_df['mm_'+str(k)].abs()
                normalized_lob_df['label_'+str(k)] = pd.cut(lt_k, ranges, right = False, labels = [-1, 0, 1])
                # draw_bar_chart(normalized_lob_df['label_'+str(k)], 'lt_%d_alpha_%f_%s.png' % (k, alpha, date))
                # draw_segmented_line_chart(normalized_lob_df[['mid_price', 'label_'+str(k)]], k, 'mid_price_seg_by_label_%d_%s.png' % (k, date))
