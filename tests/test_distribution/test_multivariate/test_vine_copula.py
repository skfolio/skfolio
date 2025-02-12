import time

import numpy as np
import pytest
import sklearn as sk

from skfolio.distribution import (
    CopulaRotation,
    Gaussian,
    GaussianCopula,
    IndependentCopula,
    StudentTCopula,
    VineCopula,
)


@pytest.fixture
def expected_marginals():
    return [
        {
            "name": "NormalInverseGaussian",
            "params": {
                "scale_": 0.01393,
                "loc_": 0.00138,
                "a_": 0.57973,
                "b_": -0.01382,
            },
        },
        {
            "name": "NormalInverseGaussian",
            "params": {
                "scale_": 0.02808,
                "loc_": -0.00079,
                "a_": 0.60408,
                "b_": 0.05767,
            },
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.01284, "loc_": 0.00054, "dof_": 3.32409},
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.01529, "loc_": 0.00149, "dof_": 3.2509},
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.01106, "loc_": 0.00044, "dof_": 2.91325},
        },
        {
            "name": "NormalInverseGaussian",
            "params": {
                "scale_": 0.01265,
                "loc_": -0.00082,
                "a_": 0.32606,
                "b_": 0.01859,
            },
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.00942, "loc_": 0.00119, "dof_": 3.1672},
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.00722, "loc_": 0.00062, "dof_": 3.21906},
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.01055, "loc_": 0.00053, "dof_": 3.01599},
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.00685, "loc_": 0.00074, "dof_": 2.88268},
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.01011, "loc_": 0.00084, "dof_": 3.0279},
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.00894, "loc_": 0.00055, "dof_": 3.37539},
        },
        {
            "name": "NormalInverseGaussian",
            "params": {
                "scale_": 0.01217,
                "loc_": 0.0011,
                "a_": 0.51399,
                "b_": -0.00267,
            },
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.00709, "loc_": 0.00071, "dof_": 3.23259},
        },
        {
            "name": "NormalInverseGaussian",
            "params": {
                "scale_": 0.01051,
                "loc_": -0.00012,
                "a_": 0.55356,
                "b_": 0.03247,
            },
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.0071, "loc_": 0.0006, "dof_": 3.09206},
        },
        {
            "name": "NormalInverseGaussian",
            "params": {
                "scale_": 0.03876,
                "loc_": -0.00549,
                "a_": 1.07205,
                "b_": 0.15593,
            },
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.01012, "loc_": 0.00093, "dof_": 3.15962},
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.00784, "loc_": 0.00052, "dof_": 3.03197},
        },
        {
            "name": "NormalInverseGaussian",
            "params": {"scale_": 0.01207, "loc_": -3e-05, "a_": 0.47691, "b_": 0.01545},
        },
    ]


@pytest.fixture
def expected_trees():
    return [
        [
            (0, 12, set(), "StudentTCopula", (0.658, 2.694)),
            (1, 12, set(), "GumbelCopula", (1.471, CopulaRotation.R180)),
            (2, 8, set(), "StudentTCopula", (0.898, 3.068)),
            (3, 6, set(), "StudentTCopula", (0.553, 4.8)),
            (4, 8, set(), "StudentTCopula", (0.546, 3.961)),
            (4, 19, set(), "StudentTCopula", (0.853, 3.434)),
            (5, 8, set(), "StudentTCopula", (0.557, 3.477)),
            (6, 8, set(), "StudentTCopula", (0.457, 3.714)),
            (6, 12, set(), "StudentTCopula", (0.509, 3.84)),
            (6, 18, set(), "StudentTCopula", (0.454, 3.78)),
            (7, 11, set(), "StudentTCopula", (0.577, 4.002)),
            (7, 15, set(), "StudentTCopula", (0.517, 3.898)),
            (7, 17, set(), "StudentTCopula", (0.48, 3.838)),
            (9, 13, set(), "StudentTCopula", (0.745, 3.431)),
            (10, 14, set(), "StudentTCopula", (0.553, 3.817)),
            (11, 14, set(), "StudentTCopula", (0.598, 4.884)),
            (13, 15, set(), "StudentTCopula", (0.675, 3.75)),
            (15, 18, set(), "StudentTCopula", (0.472, 4.93)),
            (16, 19, set(), "StudentTCopula", (0.495, 5.538)),
        ],
        [
            (0, 1, {12}, "StudentTCopula", (0.215, 12.74)),
            (0, 6, {12}, "StudentTCopula", (0.181, 18.811)),
            (2, 5, {8}, "GaussianCopula", 0.11),
            (3, 8, {6}, "StudentTCopula", (0.26, 11.079)),
            (8, 19, {4}, "StudentTCopula", (0.198, 9.216)),
            (4, 5, {8}, "StudentTCopula", (0.27, 11.02)),
            (4, 6, {8}, "StudentTCopula", (0.113, 7.572)),
            (4, 16, {19}, "StudentTCopula", (0.155, 15.946)),
            (8, 12, {6}, "StudentTCopula", (0.275, 4.513)),
            (12, 18, {6}, "StudentTCopula", (0.165, 8.776)),
            (6, 15, {18}, "StudentTCopula", (0.264, 6.968)),
            (11, 17, {7}, "StudentTCopula", (0.215, 7.467)),
            (7, 14, {11}, "StudentTCopula", (0.326, 7.367)),
            (15, 17, {7}, "StudentTCopula", (0.176, 11.657)),
            (7, 13, {15}, "StudentTCopula", (0.276, 8.221)),
            (9, 15, {13}, "StudentTCopula", (0.286, 7.061)),
            (10, 11, {14}, "StudentTCopula", (0.35, 10.226)),
            (13, 18, {15}, "StudentTCopula", (0.238, 10.944)),
        ],
        [
            (1, 6, {0, 12}, "GaussianCopula", 0.084),
            (0, 8, {12, 6}, "GumbelCopula", (1.115, CopulaRotation.R180)),
            (2, 4, {8, 5}, "StudentTCopula", (0.067, 13.471)),
            (3, 4, {8, 6}, "StudentTCopula", (0.082, 21.592)),
            (19, 5, {8, 4}, "StudentTCopula", (0.139, 15.838)),
            (8, 16, {19, 4}, "IndependentCopula", None),
            (5, 6, {8, 4}, "StudentTCopula", (0.087, 13.383)),
            (4, 12, {8, 6}, "StudentTCopula", (0.117, 12.388)),
            (8, 18, {12, 6}, "StudentTCopula", (0.065, 17.204)),
            (12, 15, {18, 6}, "StudentTCopula", (0.231, 8.519)),
            (6, 13, {18, 15}, "StudentTCopula", (0.187, 8.708)),
            (17, 14, {11, 7}, "StudentTCopula", (0.183, 12.268)),
            (11, 15, {17, 7}, "StudentTCopula", (0.141, 13.679)),
            (7, 10, {11, 14}, "StudentTCopula", (0.277, 18.333)),
            (17, 13, {7, 15}, "GumbelCopula", (1.079, CopulaRotation.R180)),
            (7, 9, {13, 15}, "StudentTCopula", (0.107, 10.433)),
            (7, 18, {13, 15}, "StudentTCopula", (0.149, 14.702)),
        ],
        [
            (1, 8, {0, 12, 6}, "GaussianCopula", 0.069),
            (0, 18, {8, 12, 6}, "GaussianCopula", 0.057),
            (2, 6, {8, 4, 5}, "IndependentCopula", None),
            (3, 5, {8, 4, 6}, "GaussianCopula", 0.071),
            (3, 12, {8, 4, 6}, "ClaytonCopula", (0.069, CopulaRotation.R180)),
            (5, 16, {8, 19, 4}, "StudentTCopula", (0.065, 16.569)),
            (19, 6, {8, 4, 5}, "IndependentCopula", None),
            (4, 18, {8, 12, 6}, "StudentTCopula", (0.067, 12.043)),
            (8, 15, {18, 12, 6}, "StudentTCopula", (0.069, 10.409)),
            (12, 13, {18, 6, 15}, "StudentTCopula", (0.17, 12.883)),
            (6, 7, {18, 13, 15}, "StudentTCopula", (0.157, 13.95)),
            (14, 15, {17, 11, 7}, "StudentTCopula", (0.082, 17.744)),
            (17, 10, {11, 14, 7}, "StudentTCopula", (0.132, 24.743)),
            (11, 13, {17, 7, 15}, "GumbelCopula", (1.055, CopulaRotation.R180)),
            (17, 9, {7, 13, 15}, "StudentTCopula", (0.059, 13.694)),
            (17, 18, {7, 13, 15}, "StudentTCopula", (0.136, 16.533)),
        ],
        [
            (1, 18, {0, 8, 12, 6}, "IndependentCopula", None),
            (0, 4, {8, 18, 12, 6}, "IndependentCopula", None),
            (2, 3, {8, 4, 5, 6}, "StudentTCopula", (0.075, 19.03)),
            (2, 19, {8, 4, 5, 6}, "IndependentCopula", None),
            (5, 12, {8, 3, 4, 6}, "StudentTCopula", (0.049, 25.837)),
            (3, 18, {8, 4, 12, 6}, "StudentTCopula", (0.107, 13.47)),
            (16, 6, {8, 19, 4, 5}, "StudentTCopula", (-0.044, 17.083)),
            (4, 15, {8, 18, 12, 6}, "StudentTCopula", (0.057, 11.575)),
            (8, 13, {18, 12, 6, 15}, "IndependentCopula", None),
            (12, 7, {18, 13, 6, 15}, "StudentTCopula", (0.138, 23.235)),
            (6, 17, {18, 15, 13, 7}, "StudentTCopula", (0.231, 20.335)),
            (15, 10, {17, 11, 14, 7}, "IndependentCopula", None),
            (14, 13, {17, 11, 15, 7}, "GaussianCopula", 0.044),
            (11, 9, {17, 15, 13, 7}, "GumbelCopula", (1.032, CopulaRotation.R180)),
            (9, 18, {17, 15, 13, 7}, "IndependentCopula", None),
        ],
        [
            (1, 4, {0, 6, 8, 12, 18}, "GaussianCopula", 0.073),
            (0, 3, {4, 6, 8, 12, 18}, "GaussianCopula", 0.137),
            (3, 19, {2, 4, 5, 6, 8}, "IndependentCopula", None),
            (2, 12, {3, 4, 5, 6, 8}, "IndependentCopula", None),
            (2, 16, {4, 5, 6, 8, 19}, "GaussianCopula", 0.098),
            (5, 18, {3, 4, 6, 8, 12}, "StudentTCopula", (0.052, 28.498)),
            (3, 15, {4, 6, 8, 12, 18}, "StudentTCopula", (-0.058, 24.473)),
            (4, 13, {6, 8, 12, 15, 18}, "StudentTCopula", (0.059, 25.309)),
            (8, 7, {6, 12, 13, 15, 18}, "StudentTCopula", (0.158, 16.693)),
            (12, 17, {6, 7, 13, 15, 18}, "GaussianCopula", 0.182),
            (6, 9, {7, 13, 15, 17, 18}, "GumbelCopula", (1.03, CopulaRotation.R0)),
            (10, 13, {7, 11, 14, 15, 17}, "IndependentCopula", None),
            (14, 9, {7, 11, 13, 15, 17}, "IndependentCopula", None),
            (11, 18, {7, 9, 13, 15, 17}, "IndependentCopula", None),
        ],
        [
            (1, 3, {0, 4, 6, 8, 12, 18}, "StudentTCopula", (0.099, 25.232)),
            (0, 5, {3, 4, 6, 8, 12, 18}, "StudentTCopula", (0.093, 30.812)),
            (19, 12, {2, 3, 4, 5, 6, 8}, "IndependentCopula", None),
            (3, 16, {2, 4, 5, 6, 8, 19}, "GaussianCopula", 0.091),
            (2, 18, {3, 4, 5, 6, 8, 12}, "IndependentCopula", None),
            (5, 15, {3, 4, 6, 8, 12, 18}, "IndependentCopula", None),
            (3, 13, {4, 6, 8, 12, 15, 18}, "GaussianCopula", -0.056),
            (4, 7, {6, 8, 12, 13, 15, 18}, "GumbelCopula", (1.054, CopulaRotation.R0)),
            (8, 17, {6, 7, 12, 13, 15, 18}, "StudentTCopula", (0.217, 10.764)),
            (12, 9, {6, 7, 13, 15, 17, 18}, "IndependentCopula", None),
            (6, 11, {7, 9, 13, 15, 17, 18}, "IndependentCopula", None),
            (10, 9, {7, 11, 13, 14, 15, 17}, "IndependentCopula", None),
            (14, 18, {7, 9, 11, 13, 15, 17}, "IndependentCopula", None),
        ],
        [
            (1, 5, {0, 3, 4, 6, 8, 12, 18}, "IndependentCopula", None),
            (0, 15, {3, 4, 5, 6, 8, 12, 18}, "IndependentCopula", None),
            (12, 16, {2, 3, 4, 5, 6, 8, 19}, "IndependentCopula", None),
            (19, 18, {2, 3, 4, 5, 6, 8, 12}, "IndependentCopula", None),
            (2, 15, {3, 4, 5, 6, 8, 12, 18}, "GaussianCopula", -0.079),
            (5, 13, {3, 4, 6, 8, 12, 15, 18}, "StudentTCopula", (0.046, 31.091)),
            (3, 7, {4, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (4, 17, {6, 7, 8, 12, 13, 15, 18}, "StudentTCopula", (0.051, 31.329)),
            (8, 9, {6, 7, 12, 13, 15, 17, 18}, "StudentTCopula", (0.153, 10.754)),
            (12, 11, {6, 7, 9, 13, 15, 17, 18}, "StudentTCopula", (0.093, 27.56)),
            (6, 14, {7, 9, 11, 13, 15, 17, 18}, "GaussianCopula", 0.088),
            (10, 18, {7, 9, 11, 13, 14, 15, 17}, "IndependentCopula", None),
        ],
        [
            (1, 15, {0, 3, 4, 5, 6, 8, 12, 18}, "GaussianCopula", -0.069),
            (0, 13, {3, 4, 5, 6, 8, 12, 15, 18}, "GaussianCopula", 0.096),
            (16, 18, {2, 3, 4, 5, 6, 8, 12, 19}, "GaussianCopula", -0.053),
            (19, 15, {2, 3, 4, 5, 6, 8, 12, 18}, "IndependentCopula", None),
            (
                2,
                13,
                {3, 4, 5, 6, 8, 12, 15, 18},
                "ClaytonCopula",
                (0.087, CopulaRotation.R90),
            ),
            (5, 7, {3, 4, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (3, 17, {4, 6, 7, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (4, 9, {6, 7, 8, 12, 13, 15, 17, 18}, "StudentTCopula", (0.15, 23.488)),
            (8, 11, {6, 7, 9, 12, 13, 15, 17, 18}, "StudentTCopula", (0.099, 32.684)),
            (12, 14, {6, 7, 9, 11, 13, 15, 17, 18}, "StudentTCopula", (0.053, 24.864)),
            (6, 10, {7, 9, 11, 13, 14, 15, 17, 18}, "IndependentCopula", None),
        ],
        [
            (1, 13, {0, 3, 4, 5, 6, 8, 12, 15, 18}, "GaussianCopula", -0.071),
            (0, 2, {3, 4, 5, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (
                16,
                15,
                {2, 3, 4, 5, 6, 8, 12, 18, 19},
                "StudentTCopula",
                (-0.055, 18.712),
            ),
            (
                19,
                13,
                {2, 3, 4, 5, 6, 8, 12, 15, 18},
                "GumbelCopula",
                (1.034, CopulaRotation.R180),
            ),
            (2, 7, {3, 4, 5, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (5, 17, {3, 4, 6, 7, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (3, 9, {4, 6, 7, 8, 12, 13, 15, 17, 18}, "GaussianCopula", -0.05),
            (
                4,
                11,
                {6, 7, 8, 9, 12, 13, 15, 17, 18},
                "StudentTCopula",
                (0.093, 24.344),
            ),
            (
                8,
                14,
                {6, 7, 9, 11, 12, 13, 15, 17, 18},
                "StudentTCopula",
                (0.082, 16.128),
            ),
            (
                12,
                10,
                {6, 7, 9, 11, 13, 14, 15, 17, 18},
                "StudentTCopula",
                (0.107, 33.14),
            ),
        ],
        [
            (
                1,
                2,
                {0, 3, 4, 5, 6, 8, 12, 13, 15, 18},
                "StudentTCopula",
                (0.086, 25.912),
            ),
            (0, 19, {2, 3, 4, 5, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (16, 13, {2, 3, 4, 5, 6, 8, 12, 15, 18, 19}, "GaussianCopula", -0.071),
            (19, 7, {2, 3, 4, 5, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (2, 17, {3, 4, 5, 6, 7, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (
                5,
                9,
                {3, 4, 6, 7, 8, 12, 13, 15, 17, 18},
                "StudentTCopula",
                (0.071, 28.547),
            ),
            (3, 11, {4, 6, 7, 8, 9, 12, 13, 15, 17, 18}, "IndependentCopula", None),
            (4, 14, {6, 7, 8, 9, 11, 12, 13, 15, 17, 18}, "IndependentCopula", None),
            (
                8,
                10,
                {6, 7, 9, 11, 12, 13, 14, 15, 17, 18},
                "StudentTCopula",
                (-0.048, 40.881),
            ),
        ],
        [
            (1, 19, {0, 2, 3, 4, 5, 6, 8, 12, 13, 15, 18}, "GaussianCopula", -0.062),
            (
                0,
                16,
                {2, 3, 4, 5, 6, 8, 12, 13, 15, 18, 19},
                "GumbelCopula",
                (1.029, CopulaRotation.R180),
            ),
            (16, 7, {2, 3, 4, 5, 6, 8, 12, 13, 15, 18, 19}, "GaussianCopula", -0.079),
            (19, 17, {2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (2, 9, {3, 4, 5, 6, 7, 8, 12, 13, 15, 17, 18}, "IndependentCopula", None),
            (5, 11, {3, 4, 6, 7, 8, 9, 12, 13, 15, 17, 18}, "IndependentCopula", None),
            (3, 14, {4, 6, 7, 8, 9, 11, 12, 13, 15, 17, 18}, "IndependentCopula", None),
            (
                4,
                10,
                {6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
        ],
        [
            (1, 16, {0, 2, 3, 4, 5, 6, 8, 12, 13, 15, 18, 19}, "GaussianCopula", 0.072),
            (
                0,
                7,
                {2, 3, 4, 5, 6, 8, 12, 13, 15, 16, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                16,
                17,
                {2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 18, 19},
                "StudentTCopula",
                (-0.044, 25.358),
            ),
            (
                19,
                9,
                {2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
            (
                2,
                11,
                {3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
            (
                5,
                14,
                {3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 17, 18},
                "StudentTCopula",
                (0.055, 20.811),
            ),
            (
                3,
                10,
                {4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
        ],
        [
            (
                1,
                7,
                {0, 2, 3, 4, 5, 6, 8, 12, 13, 15, 16, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                0,
                17,
                {2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 16, 18, 19},
                "GaussianCopula",
                0.064,
            ),
            (
                16,
                9,
                {2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 17, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                19,
                11,
                {2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
            (
                2,
                14,
                {3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
            (
                5,
                10,
                {3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
        ],
        [
            (
                1,
                17,
                {0, 2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 16, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                0,
                9,
                {2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 16, 17, 18, 19},
                "GumbelCopula",
                (1.033, CopulaRotation.R270),
            ),
            (
                16,
                11,
                {2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 17, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                19,
                14,
                {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
            (
                2,
                10,
                {3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
        ],
        [
            (
                1,
                9,
                {0, 2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 16, 17, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                0,
                11,
                {2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                16,
                14,
                {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 18, 19},
                "GaussianCopula",
                0.053,
            ),
            (
                19,
                10,
                {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18},
                "IndependentCopula",
                None,
            ),
        ],
        [
            (
                1,
                11,
                {0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                0,
                14,
                {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                16,
                10,
                {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19},
                "IndependentCopula",
                None,
            ),
        ],
        [
            (
                1,
                14,
                {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19},
                "IndependentCopula",
                None,
            ),
            (
                0,
                10,
                {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                "IndependentCopula",
                None,
            ),
        ],
        [
            (
                1,
                10,
                {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                "IndependentCopula",
                None,
            )
        ],
    ]


def _check_vine_output(model, expected_marginals, expected_trees):
    for i, marginal in enumerate(model.marginal_distributions_):
        assert type(marginal).__name__ == expected_marginals[i]["name"]
        for key, val in expected_marginals[i]["params"].items():
            assert np.isclose(getattr(marginal, key), val, atol=1e-3)

    for i, tree in enumerate(model.trees_):
        assert tree.level == i
        assert len(tree.nodes) == 20 - i
        assert len(tree.edges) == 20 - i - 1

    for i, tree in enumerate(model.trees_):
        for j, edge in enumerate(tree.edges):
            expected_edge = expected_trees[i][j]

            assert edge.cond_sets.conditioned == (expected_edge[0], expected_edge[1])
            assert edge.cond_sets.conditioning == expected_edge[2]
            assert type(edge.copula).__name__ == expected_edge[3]

            expected_param = expected_edge[4]
            if isinstance(edge.copula, StudentTCopula):
                assert np.isclose(edge.copula.rho_, expected_param[0], 1e-2)
                assert np.isclose(edge.copula.dof_, expected_param[1], 1e-2)
            elif isinstance(edge.copula, GaussianCopula):
                assert np.isclose(edge.copula.rho_, expected_param, 1e-2)
            elif isinstance(edge.copula, IndependentCopula):
                pass
            else:
                assert np.isclose(edge.copula.theta_, expected_param[0], 1e-2)
                assert edge.copula.rotation_ == expected_param[1]


def test_vine_copula(X, expected_marginals, expected_trees):
    s = time.time()
    model = VineCopula(n_jobs=-1, max_depth=100)
    model.fit(X)
    e = time.time()
    print(e - s)
    # 18s
    _check_vine_output(model, expected_marginals, expected_trees)


@pytest.mark.parametrize(
    "max_depth,central_assets,expected_trees",
    [
        (
            3,
            [True] * 1 + [False] * 19,
            [
                [
                    (0, 1, set(), "StudentTCopula", (0.465, 5.289)),
                    (0, 2, set(), "StudentTCopula", (0.373, 3.666)),
                    (0, 3, set(), "StudentTCopula", (0.39, 7.057)),
                    (0, 4, set(), "StudentTCopula", (0.295, 4.04)),
                    (0, 5, set(), "StudentTCopula", (0.326, 3.637)),
                    (0, 6, set(), "StudentTCopula", (0.435, 3.535)),
                    (0, 7, set(), "StudentTCopula", (0.306, 5.715)),
                    (0, 8, set(), "StudentTCopula", (0.413, 3.396)),
                    (0, 9, set(), "StudentTCopula", (0.307, 3.931)),
                    (0, 10, set(), "StudentTCopula", (0.286, 5.434)),
                    (0, 11, set(), "StudentTCopula", (0.267, 5.65)),
                    (0, 12, set(), "StudentTCopula", (0.658, 2.694)),
                    (0, 13, set(), "StudentTCopula", (0.369, 3.944)),
                    (0, 14, set(), "StudentTCopula", (0.313, 4.531)),
                    (0, 15, set(), "StudentTCopula", (0.311, 4.393)),
                    (0, 16, set(), "StudentTCopula", (0.184, 6.641)),
                    (0, 17, set(), "StudentTCopula", (0.369, 5.057)),
                    (0, 18, set(), "StudentTCopula", (0.304, 5.532)),
                    (0, 19, set(), "StudentTCopula", (0.3, 4.098)),
                ],
                [
                    (1, 12, {0}, "StudentTCopula", (0.272, 9.648)),
                    (2, 8, {0}, "StudentTCopula", (0.875, 3.412)),
                    (3, 6, {0}, "StudentTCopula", (0.463, 7.253)),
                    (4, 8, {0}, "StudentTCopula", (0.48, 5.598)),
                    (4, 19, {0}, "StudentTCopula", (0.837, 3.989)),
                    (5, 8, {0}, "StudentTCopula", (0.484, 5.209)),
                    (6, 8, {0}, "StudentTCopula", (0.342, 6.512)),
                    (6, 12, {0}, "StudentTCopula", (0.327, 11.048)),
                    (6, 18, {0}, "StudentTCopula", (0.372, 5.069)),
                    (7, 11, {0}, "StudentTCopula", (0.547, 5.166)),
                    (7, 15, {0}, "StudentTCopula", (0.46, 5.221)),
                    (7, 17, {0}, "StudentTCopula", (0.416, 5.32)),
                    (9, 13, {0}, "StudentTCopula", (0.711, 4.158)),
                    (10, 14, {0}, "StudentTCopula", (0.507, 5.089)),
                    (11, 14, {0}, "StudentTCopula", (0.56, 6.337)),
                    (13, 15, {0}, "StudentTCopula", (0.629, 4.539)),
                    (15, 18, {0}, "StudentTCopula", (0.414, 7.814)),
                    (16, 19, {0}, "StudentTCopula", (0.468, 6.886)),
                ],
                [
                    (1, 6, {0, 12}, "StudentTCopula", (0.091, 32.93)),
                    (2, 5, {8, 0}, "GaussianCopula", 0.114),
                    (3, 8, {0, 6}, "StudentTCopula", (0.222, 13.849)),
                    (3, 18, {0, 6}, "StudentTCopula", (0.112, 11.045)),
                    (8, 19, {0, 4}, "StudentTCopula", (0.181, 12.907)),
                    (4, 5, {8, 0}, "StudentTCopula", (0.262, 15.856)),
                    (4, 6, {8, 0}, "StudentTCopula", (0.083, 9.51)),
                    (4, 16, {0, 19}, "StudentTCopula", (0.152, 30.607)),
                    (8, 12, {0, 6}, "StudentTCopula", (0.17, 12.729)),
                    (6, 15, {0, 18}, "StudentTCopula", (0.203, 9.685)),
                    (11, 15, {0, 7}, "StudentTCopula", (0.157, 12.114)),
                    (11, 17, {0, 7}, "StudentTCopula", (0.189, 8.634)),
                    (7, 14, {0, 11}, "StudentTCopula", (0.297, 9.522)),
                    (7, 13, {0, 15}, "StudentTCopula", (0.243, 9.174)),
                    (9, 15, {0, 13}, "StudentTCopula", (0.279, 7.941)),
                    (10, 11, {0, 14}, "StudentTCopula", (0.34, 10.929)),
                    (13, 18, {0, 15}, "StudentTCopula", (0.201, 14.006)),
                ],
            ],
        ),
        (
            4,
            [False] + [True] * 2 + [False] * 17,
            [
                [
                    (0, 1, set(), "StudentTCopula", (0.465, 5.289)),
                    (1, 2, set(), "GumbelCopula", (1.255, CopulaRotation.R180)),
                    (1, 12, set(), "GumbelCopula", (1.471, CopulaRotation.R180)),
                    (2, 3, set(), "StudentTCopula", (0.417, 5.034)),
                    (2, 4, set(), "StudentTCopula", (0.519, 4.74)),
                    (2, 5, set(), "StudentTCopula", (0.538, 4.13)),
                    (2, 6, set(), "StudentTCopula", (0.409, 4.362)),
                    (2, 7, set(), "StudentTCopula", (0.293, 4.18)),
                    (2, 8, set(), "StudentTCopula", (0.898, 3.068)),
                    (2, 9, set(), "StudentTCopula", (0.304, 3.423)),
                    (2, 10, set(), "StudentTCopula", (0.241, 4.888)),
                    (2, 11, set(), "StudentTCopula", (0.294, 5.032)),
                    (2, 13, set(), "StudentTCopula", (0.224, 3.986)),
                    (2, 14, set(), "StudentTCopula", (0.326, 3.833)),
                    (2, 15, set(), "StudentTCopula", (0.23, 3.408)),
                    (2, 16, set(), "StudentTCopula", (0.332, 7.68)),
                    (2, 17, set(), "StudentTCopula", (0.401, 4.074)),
                    (2, 18, set(), "StudentTCopula", (0.244, 4.603)),
                    (2, 19, set(), "StudentTCopula", (0.52, 4.369)),
                ],
                [
                    (0, 2, {1}, "StudentTCopula", (0.29, 6.776)),
                    (2, 12, {1}, "StudentTCopula", (0.313, 5.819)),
                    (1, 3, {2}, "StudentTCopula", (0.221, 12.524)),
                    (1, 4, {2}, "StudentTCopula", (0.121, 18.549)),
                    (1, 5, {2}, "StudentTCopula", (0.093, 17.375)),
                    (1, 6, {2}, "StudentTCopula", (0.234, 13.817)),
                    (1, 7, {2}, "GumbelCopula", (1.058, CopulaRotation.R180)),
                    (1, 8, {2}, "StudentTCopula", (0.049, 18.054)),
                    (1, 9, {2}, "StudentTCopula", (0.069, 21.885)),
                    (1, 10, {2}, "StudentTCopula", (0.118, 16.544)),
                    (1, 11, {2}, "GaussianCopula", 0.069),
                    (1, 13, {2}, "StudentTCopula", (0.108, 19.567)),
                    (1, 14, {2}, "StudentTCopula", (0.126, 18.429)),
                    (1, 15, {2}, "StudentTCopula", (0.096, 18.208)),
                    (1, 16, {2}, "GaussianCopula", 0.103),
                    (1, 17, {2}, "GaussianCopula", 0.141),
                    (1, 18, {2}, "StudentTCopula", (0.129, 22.901)),
                    (1, 19, {2}, "StudentTCopula", (0.064, 28.687)),
                ],
                [
                    (0, 12, {1, 2}, "StudentTCopula", (0.539, 5.419)),
                    (12, 13, {1, 2}, "StudentTCopula", (0.383, 7.587)),
                    (3, 6, {1, 2}, "StudentTCopula", (0.43, 7.936)),
                    (4, 9, {1, 2}, "StudentTCopula", (0.23, 6.499)),
                    (4, 19, {1, 2}, "StudentTCopula", (0.793, 3.501)),
                    (5, 19, {1, 2}, "StudentTCopula", (0.317, 6.828)),
                    (6, 8, {1, 2}, "StudentTCopula", (0.235, 13.863)),
                    (6, 18, {1, 2}, "StudentTCopula", (0.395, 5.772)),
                    (7, 11, {1, 2}, "StudentTCopula", (0.536, 5.206)),
                    (7, 13, {1, 2}, "StudentTCopula", (0.482, 6.54)),
                    (7, 17, {1, 2}, "StudentTCopula", (0.413, 5.757)),
                    (9, 13, {1, 2}, "StudentTCopula", (0.724, 4.29)),
                    (10, 14, {1, 2}, "StudentTCopula", (0.51, 5.122)),
                    (11, 14, {1, 2}, "StudentTCopula", (0.545, 6.659)),
                    (13, 15, {1, 2}, "StudentTCopula", (0.647, 4.994)),
                    (15, 18, {1, 2}, "StudentTCopula", (0.435, 7.805)),
                    (16, 19, {1, 2}, "StudentTCopula", (0.405, 8.859)),
                ],
                [
                    (0, 13, {1, 2, 12}, "StudentTCopula", (0.17, 18.531)),
                    (12, 7, {1, 2, 13}, "StudentTCopula", (0.165, 19.068)),
                    (3, 18, {1, 2, 6}, "StudentTCopula", (0.113, 11.769)),
                    (9, 19, {1, 2, 4}, "StudentTCopula", (0.079, 12.275)),
                    (4, 13, {9, 2, 1}, "IndependentCopula", None),
                    (4, 5, {1, 2, 19}, "StudentTCopula", (0.071, 32.705)),
                    (4, 16, {1, 2, 19}, "StudentTCopula", (0.133, 28.982)),
                    (8, 18, {1, 2, 6}, "ClaytonCopula", (0.092, CopulaRotation.R0)),
                    (6, 15, {1, 18, 2}, "StudentTCopula", (0.229, 13.208)),
                    (11, 13, {1, 2, 7}, "StudentTCopula", (0.175, 15.779)),
                    (11, 17, {1, 2, 7}, "StudentTCopula", (0.174, 9.339)),
                    (7, 14, {1, 2, 11}, "StudentTCopula", (0.304, 10.263)),
                    (7, 15, {1, 2, 13}, "StudentTCopula", (0.257, 9.957)),
                    (9, 15, {1, 2, 13}, "StudentTCopula", (0.27, 8.792)),
                    (10, 11, {1, 2, 14}, "StudentTCopula", (0.342, 11.775)),
                    (13, 18, {1, 2, 15}, "StudentTCopula", (0.228, 14.134)),
                ],
            ],
        ),
    ],
)
def test_vine_copula_truncated_central_assets_output(
    X, expected_marginals, max_depth, central_assets, expected_trees
):
    model = VineCopula(n_jobs=-1, max_depth=max_depth, central_assets=central_assets)
    model.fit(X)
    _check_vine_output(model, expected_marginals, expected_trees)
    _ = model.sample(n_samples=4)


def test_vine_copula_conditional_sampling(X):
    model = VineCopula(
        n_jobs=-1, max_depth=3, central_assets=[False] + [True] * 2 + [False] * 17
    )
    model.fit(X)
    sample = model.sample(
        n_samples=4,
        random_state=42,
        conditioning_samples={
            0: [-0.1, -0.2, -0.3, -0.4],
            1: [-0.2, -0.3, -0.4, -0.5],
        },
    )
    np.testing.assert_array_almost_equal(
        sample,
        [
            [
                -0.1,
                -0.2,
                -0.09255,
                -0.04748,
                0.00458,
                -0.09544,
                -0.02684,
                -0.02331,
                -0.09718,
                0.02677,
                -0.06239,
                -0.06346,
                -0.00236,
                0.03242,
                -0.0457,
                0.03712,
                -0.02105,
                -0.0707,
                -0.02595,
                -0.06537,
            ],
            [
                -0.2,
                -0.3,
                -0.19973,
                -0.04873,
                -0.05702,
                0.06058,
                -0.10032,
                -0.09621,
                -0.08966,
                -0.14291,
                -0.19791,
                -0.06049,
                -0.18053,
                -0.06708,
                -0.09156,
                -0.05021,
                -0.04647,
                -0.26097,
                -0.06194,
                -0.04821,
            ],
            [
                -0.3,
                -0.4,
                -0.60121,
                -0.02089,
                -0.04442,
                0.06281,
                -0.18503,
                -0.54534,
                -0.47727,
                -0.48239,
                -1.11775,
                -0.81655,
                -0.20855,
                -0.37635,
                -0.18034,
                -0.23251,
                -0.09168,
                0.04289,
                -0.14937,
                -0.09728,
            ],
            [
                -0.4,
                -0.5,
                -0.675,
                -1.22643,
                -0.72512,
                -0.18406,
                -0.60088,
                -0.62511,
                -0.08972,
                -0.05973,
                0.01138,
                -0.27297,
                -0.2116,
                0.16265,
                -0.08687,
                -0.23019,
                -0.27888,
                -1.0494,
                -0.30444,
                -0.19437,
            ],
        ],
        5,
    )


@pytest.mark.parametrize("max_depth", list(range(2, 22)))
def test_vine_truncated_sampling_order(X, max_depth):
    model = VineCopula(
        max_depth=max_depth,
        marginal_candidates=[Gaussian()],
        copula_candidates=[GaussianCopula()],
        n_jobs=-1,
    )
    model.fit(X)
    samples = model.sample(n_samples=4, random_state=42)
    assert samples.shape == (4, 20)

    sampling_order = model._sampling_order()

    lvars = []
    for node, is_left in sampling_order[::-1]:
        i = 0 if is_left else 1
        if isinstance(node.ref, int):
            for var in lvars:
                assert var != node.ref
            lvars.append(node.ref)
        else:
            for var in lvars:
                assert var not in node.ref.cond_sets.conditioned
            lvars.append(node.ref.cond_sets.conditioned[i])


@pytest.mark.parametrize(
    "max_depth,expected",
    [
        (
            100,
            [
                [
                    0.00722,
                    0.00191,
                    -0.00776,
                    0.02012,
                    -0.00957,
                    -0.0344,
                    0.01573,
                    0.007,
                    -0.00317,
                    0.00615,
                    0.00517,
                    0.00969,
                    0.0132,
                    -0.00566,
                    -0.00175,
                    -0.0008,
                    -0.03785,
                    0.00728,
                    0.00784,
                    -0.01681,
                ],
                [
                    -0.0182,
                    -0.01502,
                    0.01463,
                    -0.02521,
                    0.00688,
                    0.00248,
                    -0.00686,
                    -0.00233,
                    0.00381,
                    -0.00671,
                    -0.00014,
                    -0.01146,
                    -0.00636,
                    -0.00686,
                    0.00052,
                    -0.00328,
                    0.00449,
                    -0.00026,
                    -0.00481,
                    0.01691,
                ],
            ],
        ),
        (
            4,
            [
                [
                    -0.00005,
                    -0.00348,
                    -0.00129,
                    0.02036,
                    0.01266,
                    0.03857,
                    0.01579,
                    -0.0078,
                    -0.00177,
                    -0.00286,
                    -0.01201,
                    -0.02211,
                    0.01178,
                    0.00013,
                    -0.00008,
                    0.00695,
                    0.01579,
                    -0.00041,
                    -0.00223,
                    0.00547,
                ],
                [
                    0.0141,
                    -0.00349,
                    -0.00361,
                    -0.03154,
                    -0.00304,
                    -0.01556,
                    -0.0069,
                    -0.00582,
                    -0.00515,
                    0.00315,
                    -0.01009,
                    -0.00136,
                    -0.00644,
                    0.00121,
                    -0.01392,
                    -0.00365,
                    -0.03682,
                    -0.00002,
                    0.00295,
                    0.00511,
                ],
            ],
        ),
    ],
)
def test_vine_sample(X, max_depth, expected):
    model = VineCopula(max_depth=max_depth, n_jobs=-1)
    model.fit(X)
    sample = model.sample(n_samples=2, random_state=42)
    np.testing.assert_array_almost_equal(sample, expected, 5)


def test_vine_sample_truncated_consistency(X):
    ref_model = VineCopula(max_depth=10)
    ref_model.fit(np.array(X)[:, :5])
    dummy_X = np.ones((len(X), 2)) / 2
    for i in [1, 2]:
        for edge in ref_model.trees_[-i].edges:
            edge.copula = IndependentCopula().fit(dummy_X)
    ref_samples = ref_model.sample(n_samples=4, random_state=42)

    model = VineCopula(max_depth=2)
    model.fit(np.array(X)[:, :5])
    samples = model.sample(n_samples=4, random_state=42)

    np.testing.assert_almost_equal(samples, ref_samples)


def _test_fit_re_fit(X):
    model = VineCopula(
        max_depth=100,
        marginal_candidates=[Gaussian()],
        copula_candidates=[GaussianCopula()],
        n_jobs=-1,
        independence_level=1.5,
    )
    model.fit(X)
    sample = model.sample(n_samples=int(5e5), random_state=42)
    model.fit(sample)
    sample = model.sample(n_samples=int(5e5), random_state=42)
    model3 = sk.clone(model)
    model3.fit(sample)
    np.testing.assert_array_almost_equal(
        model.sample(n_samples=5, random_state=42),
        model3.sample(n_samples=5, random_state=42),
    )


def _generate_checks_marginals(model):
    params = ["scale_", "loc_", "dof_", "a_", "b_"]
    res = []
    for dist in model.marginal_distributions_:
        res.append(
            {
                "name": type(dist).__name__,
                "params": {
                    param: round(float(getattr(dist, param)), 5)
                    for param in params
                    if hasattr(dist, param)
                },
            }
        )
    return res


def _generate_checks(model):
    trees = []
    for tree in model.trees_:
        edges = []
        for edge in tree.edges:
            if type(edge.copula).__name__ == "StudentTCopula":
                params = (
                    round(float(edge.copula.rho_), 3),
                    round(float(edge.copula.dof_), 3),
                )
            elif type(edge.copula).__name__ == "GaussianCopula":
                params = round(float(edge.copula.rho_), 3)
            elif type(edge.copula).__name__ == "IndependentCopula":
                params = None
            else:
                params = round(float(edge.copula.theta_), 3), edge.copula.rotation_
            edges.append(
                (
                    edge.cond_sets.conditioned[0],
                    edge.cond_sets.conditioned[1],
                    edge.cond_sets.conditioning,
                    type(edge.copula).__name__,
                    params,
                )
            )
        trees.append(edges)
    return trees
