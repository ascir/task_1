import scipy.stats as stats


def t_test(a, b, alt):
    return stats.ttest_ind(a, b,alternative = alt)    