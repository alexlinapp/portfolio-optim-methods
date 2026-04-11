import numpy as np

from portfolio_optim.portfolio.covariance import low_rank_psd, sample_covariance


def test_low_rank_psd_is_symmetric_and_pd_toy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 10))
    s = sample_covariance(x)
    lr = low_rank_psd(s, rank=3, ridge=1e-6)
    assert np.allclose(lr, lr.T)
    w = np.linalg.eigvalsh(lr)
    assert np.min(w) > 0
