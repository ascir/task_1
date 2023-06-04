import scipy.stats as stats


def t_test(a, b, alt = 'less'):
    return stats.ttest_ind(a, b,alternative = alt)    