import pymc3 as pm
import theano.tensor as tt


def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?

    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)


def full(model, ic_prior, name, gmt, xfa, xfb, ic_mu=0.0, ic_sigma=1.0, ic_fac=1.0):

    with model:

        intercept = ic_fac * ic_prior(name + "_intercept", mu=ic_mu, sigma=ic_sigma)

        slope = pm.Normal(name + "_slope", mu=0, sigma=1)
        yearly = pm.Normal(name + "_yearly", mu=0.0, sd=5.0, shape=xfa.dshape[1])
        trend = pm.Normal(name + "_trend", mu=0.0, sd=2.0, shape=xfb.dshape[1])

        param = intercept / (
            1
            + tt.exp(
                -1 * (slope * gmt + det_dot(xfa, yearly) + gmt * det_dot(xfb, trend))
            )
        )

    return pm.Deterministic(name, param)


def longterm_yearlycycle(model, intercept, name, gmt, xfa):

    with model:

        # intercept = ic_fac*ic_prior(name + "_intercept", mu=ic_mu, sigma=ic_sigma)

        slope = pm.Normal(name + "_slope", mu=0, sigma=1)
        yearly = pm.Normal(name + "_yearly", mu=0.0, sd=5.0, shape=xfa.dshape[1])

        param = intercept / (1 + tt.exp(-1 * (slope * gmt + det_dot(xfa, yearly))))

    return pm.Deterministic(name, param)


def longterm(model, ic_prior, name, gmt, ic_mu=0.0, ic_sigma=1.0, ic_fac=1.0):

    with model:

        intercept = ic_fac * ic_prior(name + "_intercept", mu=ic_mu, sigma=ic_sigma)

        slope = pm.Normal(name + "_slope", mu=0, sigma=1)
        param = intercept / (1 + tt.exp(-1 * (slope * gmt)))

    return pm.Deterministic(name, param)


def yearlycycle(model, ic_prior, name, xfa, ic_mu=0.0, ic_sigma=1.0, ic_fac=1.0):

    with model:

        intercept = ic_fac * ic_prior(name + "_intercept", mu=ic_mu, sigma=ic_sigma)

        yearly = pm.Normal(name + "_yearly", mu=0.0, sd=5.0, shape=xfa.dshape[1])

        param = intercept / (1 + tt.exp(-1 * det_dot(xfa, yearly)))

    return pm.Deterministic(name, param)
