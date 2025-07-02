import  numpy as np
import  pandas as pd

def corrX_new(df, label, cut=0.9):
    # Get correlation matrix and upper triagle
    corr_mtx = df.corr().abs()
    avg_corr = corr_mtx.mean(axis=1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(bool))

    # Get correlation between features and label
    cross_corr = df[df.columns[0:]].corrwith(label[:])

    dropcols = list()

    columns=(['v1', 'v2', 'v1.target','v2.target', 'corr', 'drop'])

    for row in range(len(up) - 1):
        col_idx = row + 1
        for col in range(col_idx, len(up)):
            if (corr_mtx.iloc[row, col] > cut):
                if (abs(cross_corr.iloc[row]) > abs(cross_corr.iloc[col])): # Feature row has a higher correlation with the label than the feature column
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]
                else:
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]

                s = pd.Series([corr_mtx.index[row],
                               up.columns[col],
                               avg_corr[row],
                               avg_corr[col],
                               up.iloc[row, col],
                               drop],
                               index=columns)

                # res = res.append(s, ignore_index=True)
                try:
                    res = pd.concat([res, s], axis = 1, ignore_index=True)
                except:
                    res = pd.DataFrame(s)                    

    dropcols_names = calcDrop(res.T)

    return (dropcols_names)


def calcDrop(res):
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))

    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))

    # Drop any variables in same row as a keep variable
    p = res[res['v1'].isin(keep) | res['v2'].isin(keep)][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop
    poss_drop = list(set(poss_drop).difference(set(drop)))

    # subset res dataframe to include possible drop pairs
    m = res[res['v1'].isin(poss_drop) | res['v2'].isin(poss_drop)][['v1', 'v2', 'drop']]

    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)

    return drop