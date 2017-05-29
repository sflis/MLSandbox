


def pval(uplim, uplim_dist):
    from scipy import stats
    import numpy as np
    i = 0
    uplim_dist = np.sort(uplim_dist)
    while(((i+1)<len(uplim_dist)) and uplim_dist[i]< uplim):
        i +=1
    #Computing the p-value
    p_value = 1-float(i)/len(uplim_dist)
    p_value_sigma = stats.norm.ppf(1.0 - p_value)
    return (p_value,p_value_sigma)


def sigma_bands_central(lim_distr,sigma_list):
    from scipy import stats, integrate
    import numpy as np
    sigma_bands = dict()
    for j in sigma_list:
        val, err = integrate.quad(stats.norm.pdf, -(j),j)
        sigma_bands[j] = (val,
                          np.percentile(lim_distr, ((1.-val)/2)*100),
                          np.percentile(lim_distr, (val+(1.-val)/2)*100))

    return sigma_bands
