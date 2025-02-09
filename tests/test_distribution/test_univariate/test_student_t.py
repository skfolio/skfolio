from skfolio.distribution import NormalInverseGaussian


def test_student_t(returns):
    model = NormalInverseGaussian()
    model.fit(returns)
    model.score_samples(returns)
    model.score(returns)
    model.bic(returns)
    model.cdf(returns)
    model.sample(4)
