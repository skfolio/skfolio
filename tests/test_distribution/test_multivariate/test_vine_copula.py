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
            "name": "StudentT",
            "params": {"scale_": 0.02392, "loc_": 0.00093, "dof_": 3.24613},
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
            "name": "StudentT",
            "params": {"scale_": 0.01077, "loc_": 0.00112, "dof_": 3.03596},
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
            (0, 12, set(), "StudentTCopula", (0.658, 2.733)),
            (1, 12, set(), "GumbelCopula", (1.471, CopulaRotation.R180)),
            (2, 8, set(), "StudentTCopula", (0.898, 3.068)),
            (3, 6, set(), "StudentTCopula", (0.553, 4.8)),
            (4, 8, set(), "StudentTCopula", (0.546, 3.961)),
            (4, 19, set(), "StudentTCopula", (0.853, 3.434)),
            (5, 8, set(), "StudentTCopula", (0.557, 3.477)),
            (6, 8, set(), "StudentTCopula", (0.457, 3.714)),
            (6, 12, set(), "StudentTCopula", (0.509, 3.862)),
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
            (0, 1, {12}, "StudentTCopula", (0.214, 12.296)),
            (0, 6, {12}, "StudentTCopula", (0.183, 17.967)),
            (2, 5, {8}, "GaussianCopula", 0.11),
            (3, 8, {6}, "StudentTCopula", (0.26, 11.079)),
            (8, 19, {4}, "StudentTCopula", (0.198, 9.216)),
            (4, 5, {8}, "StudentTCopula", (0.27, 11.02)),
            (4, 6, {8}, "StudentTCopula", (0.113, 7.572)),
            (4, 16, {19}, "StudentTCopula", (0.155, 15.946)),
            (8, 12, {6}, "StudentTCopula", (0.275, 4.427)),
            (12, 18, {6}, "StudentTCopula", (0.166, 8.675)),
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
            (1, 6, {0, 12}, "GaussianCopula", 0.083),
            (0, 8, {12, 6}, "GumbelCopula", (1.116, CopulaRotation.R180)),
            (2, 4, {8, 5}, "StudentTCopula", (0.067, 13.471)),
            (3, 4, {8, 6}, "StudentTCopula", (0.082, 21.592)),
            (19, 5, {8, 4}, "StudentTCopula", (0.139, 15.838)),
            (8, 16, {19, 4}, "IndependentCopula", None),
            (5, 6, {8, 4}, "StudentTCopula", (0.087, 13.383)),
            (4, 12, {8, 6}, "StudentTCopula", (0.117, 11.509)),
            (8, 18, {12, 6}, "StudentTCopula", (0.066, 17.37)),
            (12, 15, {18, 6}, "StudentTCopula", (0.23, 8.539)),
            (6, 13, {18, 15}, "StudentTCopula", (0.187, 8.708)),
            (17, 14, {11, 7}, "StudentTCopula", (0.183, 12.268)),
            (11, 15, {17, 7}, "StudentTCopula", (0.141, 13.679)),
            (7, 10, {11, 14}, "StudentTCopula", (0.277, 18.333)),
            (17, 13, {7, 15}, "GumbelCopula", (1.079, CopulaRotation.R180)),
            (7, 9, {13, 15}, "StudentTCopula", (0.107, 10.433)),
            (7, 18, {13, 15}, "StudentTCopula", (0.149, 14.702)),
        ],
        [
            (1, 8, {0, 12, 6}, "ClaytonCopula", (0.092, CopulaRotation.R0)),
            (0, 18, {8, 12, 6}, "GaussianCopula", 0.057),
            (2, 6, {8, 4, 5}, "IndependentCopula", None),
            (3, 5, {8, 4, 6}, "GaussianCopula", 0.071),
            (3, 12, {8, 4, 6}, "ClaytonCopula", (0.069, CopulaRotation.R180)),
            (5, 16, {8, 19, 4}, "StudentTCopula", (0.065, 16.569)),
            (19, 6, {8, 4, 5}, "IndependentCopula", None),
            (4, 18, {8, 12, 6}, "StudentTCopula", (0.067, 12.123)),
            (8, 15, {18, 12, 6}, "StudentTCopula", (0.07, 10.382)),
            (12, 13, {18, 6, 15}, "StudentTCopula", (0.169, 13.013)),
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
            (5, 12, {8, 3, 4, 6}, "StudentTCopula", (0.049, 24.198)),
            (3, 18, {8, 4, 12, 6}, "StudentTCopula", (0.107, 13.478)),
            (16, 6, {8, 19, 4, 5}, "StudentTCopula", (-0.044, 17.083)),
            (4, 15, {8, 18, 12, 6}, "StudentTCopula", (0.057, 11.607)),
            (8, 13, {18, 12, 6, 15}, "IndependentCopula", None),
            (12, 7, {18, 13, 6, 15}, "StudentTCopula", (0.138, 21.522)),
            (6, 17, {18, 15, 13, 7}, "StudentTCopula", (0.231, 20.335)),
            (15, 10, {17, 11, 14, 7}, "IndependentCopula", None),
            (14, 13, {17, 11, 15, 7}, "GaussianCopula", 0.044),
            (11, 9, {17, 15, 13, 7}, "GumbelCopula", (1.032, CopulaRotation.R180)),
            (9, 18, {17, 15, 13, 7}, "IndependentCopula", None),
        ],
        [
            (1, 4, {0, 6, 8, 12, 18}, "GaussianCopula", 0.074),
            (0, 3, {4, 6, 8, 12, 18}, "GaussianCopula", 0.136),
            (3, 19, {2, 4, 5, 6, 8}, "IndependentCopula", None),
            (2, 12, {3, 4, 5, 6, 8}, "IndependentCopula", None),
            (2, 16, {4, 5, 6, 8, 19}, "GaussianCopula", 0.098),
            (5, 18, {3, 4, 6, 8, 12}, "StudentTCopula", (0.052, 28.764)),
            (3, 15, {4, 6, 8, 12, 18}, "StudentTCopula", (-0.059, 24.673)),
            (4, 13, {6, 8, 12, 15, 18}, "StudentTCopula", (0.059, 25.456)),
            (8, 7, {6, 12, 13, 15, 18}, "StudentTCopula", (0.158, 16.727)),
            (12, 17, {6, 7, 13, 15, 18}, "GaussianCopula", 0.182),
            (6, 9, {7, 13, 15, 17, 18}, "GumbelCopula", (1.03, CopulaRotation.R0)),
            (10, 13, {7, 11, 14, 15, 17}, "IndependentCopula", None),
            (14, 9, {7, 11, 13, 15, 17}, "IndependentCopula", None),
            (11, 18, {7, 9, 13, 15, 17}, "IndependentCopula", None),
        ],
        [
            (1, 3, {0, 4, 6, 8, 12, 18}, "ClaytonCopula", (0.134, CopulaRotation.R0)),
            (0, 5, {3, 4, 6, 8, 12, 18}, "StudentTCopula", (0.093, 31.594)),
            (19, 12, {2, 3, 4, 5, 6, 8}, "IndependentCopula", None),
            (3, 16, {2, 4, 5, 6, 8, 19}, "GaussianCopula", 0.091),
            (2, 18, {3, 4, 5, 6, 8, 12}, "IndependentCopula", None),
            (5, 15, {3, 4, 6, 8, 12, 18}, "IndependentCopula", None),
            (3, 13, {4, 6, 8, 12, 15, 18}, "GaussianCopula", -0.056),
            (4, 7, {6, 8, 12, 13, 15, 18}, "GumbelCopula", (1.054, CopulaRotation.R0)),
            (8, 17, {6, 7, 12, 13, 15, 18}, "StudentTCopula", (0.217, 10.857)),
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
            (5, 13, {3, 4, 6, 8, 12, 15, 18}, "StudentTCopula", (0.046, 31.55)),
            (3, 7, {4, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (4, 17, {6, 7, 8, 12, 13, 15, 18}, "StudentTCopula", (0.051, 31.397)),
            (8, 9, {6, 7, 12, 13, 15, 17, 18}, "StudentTCopula", (0.153, 10.689)),
            (12, 11, {6, 7, 9, 13, 15, 17, 18}, "StudentTCopula", (0.093, 26.03)),
            (6, 14, {7, 9, 11, 13, 15, 17, 18}, "GaussianCopula", 0.088),
            (10, 18, {7, 9, 11, 13, 14, 15, 17}, "IndependentCopula", None),
        ],
        [
            (1, 15, {0, 3, 4, 5, 6, 8, 12, 18}, "GaussianCopula", -0.072),
            (0, 13, {3, 4, 5, 6, 8, 12, 15, 18}, "GaussianCopula", 0.097),
            (16, 18, {2, 3, 4, 5, 6, 8, 12, 19}, "GaussianCopula", -0.054),
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
            (4, 9, {6, 7, 8, 12, 13, 15, 17, 18}, "StudentTCopula", (0.15, 23.67)),
            (8, 11, {6, 7, 9, 12, 13, 15, 17, 18}, "StudentTCopula", (0.099, 32.766)),
            (12, 14, {6, 7, 9, 11, 13, 15, 17, 18}, "StudentTCopula", (0.051, 22.947)),
            (6, 10, {7, 9, 11, 13, 14, 15, 17, 18}, "IndependentCopula", None),
        ],
        [
            (1, 13, {0, 3, 4, 5, 6, 8, 12, 15, 18}, "GaussianCopula", -0.07),
            (0, 2, {3, 4, 5, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (
                16,
                15,
                {2, 3, 4, 5, 6, 8, 12, 18, 19},
                "StudentTCopula",
                (-0.055, 18.045),
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
            (3, 9, {4, 6, 7, 8, 12, 13, 15, 17, 18}, "GaussianCopula", -0.051),
            (
                4,
                11,
                {6, 7, 8, 9, 12, 13, 15, 17, 18},
                "StudentTCopula",
                (0.093, 24.336),
            ),
            (
                8,
                14,
                {6, 7, 9, 11, 12, 13, 15, 17, 18},
                "StudentTCopula",
                (0.083, 16.145),
            ),
            (
                12,
                10,
                {6, 7, 9, 11, 13, 14, 15, 17, 18},
                "StudentTCopula",
                (0.109, 31.347),
            ),
        ],
        [
            (
                1,
                2,
                {0, 3, 4, 5, 6, 8, 12, 13, 15, 18},
                "StudentTCopula",
                (0.085, 23.337),
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
                (0.071, 28.569),
            ),
            (3, 11, {4, 6, 7, 8, 9, 12, 13, 15, 17, 18}, "IndependentCopula", None),
            (4, 14, {6, 7, 8, 9, 11, 12, 13, 15, 17, 18}, "IndependentCopula", None),
            (
                8,
                10,
                {6, 7, 9, 11, 12, 13, 14, 15, 17, 18},
                "StudentTCopula",
                (-0.048, 41.426),
            ),
        ],
        [
            (1, 19, {0, 2, 3, 4, 5, 6, 8, 12, 13, 15, 18}, "GaussianCopula", -0.06),
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
            (1, 16, {0, 2, 3, 4, 5, 6, 8, 12, 13, 15, 18, 19}, "GaussianCopula", 0.071),
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
                (-0.044, 25.346),
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
                (0.055, 20.866),
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
                0.065,
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
                (1.035, CopulaRotation.R270),
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
                "GaussianCopula",
                0.045,
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
    # _check_vine_output(model, expected_marginals, expected_trees)


@pytest.mark.parametrize(
    "max_depth,central_assets,expected_trees",
    [
        (
            3,
            [True] * 1 + [False] * 19,
            [
                [
                    (0, 1, set(), "GumbelCopula", (1.445, CopulaRotation.R180)),
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
                    (0, 12, set(), "StudentTCopula", (0.658, 2.733)),
                    (0, 13, set(), "StudentTCopula", (0.369, 3.944)),
                    (0, 14, set(), "StudentTCopula", (0.313, 4.531)),
                    (0, 15, set(), "StudentTCopula", (0.311, 4.393)),
                    (0, 16, set(), "StudentTCopula", (0.184, 6.641)),
                    (0, 17, set(), "StudentTCopula", (0.369, 5.057)),
                    (0, 18, set(), "StudentTCopula", (0.304, 5.532)),
                    (0, 19, set(), "StudentTCopula", (0.3, 4.098)),
                ],
                [
                    (1, 12, {0}, "StudentTCopula", (0.27, 9.953)),
                    (2, 8, {0}, "StudentTCopula", (0.875, 3.412)),
                    (3, 6, {0}, "StudentTCopula", (0.463, 7.253)),
                    (4, 8, {0}, "StudentTCopula", (0.48, 5.598)),
                    (4, 19, {0}, "StudentTCopula", (0.837, 3.989)),
                    (5, 8, {0}, "StudentTCopula", (0.484, 5.209)),
                    (6, 8, {0}, "StudentTCopula", (0.342, 6.512)),
                    (6, 12, {0}, "StudentTCopula", (0.326, 11.048)),
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
                    (1, 6, {0, 12}, "GumbelCopula", (1.059, CopulaRotation.R0)),
                    (2, 5, {8, 0}, "GaussianCopula", 0.114),
                    (3, 8, {0, 6}, "StudentTCopula", (0.222, 13.849)),
                    (3, 18, {0, 6}, "StudentTCopula", (0.112, 11.045)),
                    (8, 19, {0, 4}, "StudentTCopula", (0.181, 12.907)),
                    (4, 5, {8, 0}, "StudentTCopula", (0.262, 15.856)),
                    (4, 6, {8, 0}, "StudentTCopula", (0.083, 9.51)),
                    (4, 16, {0, 19}, "StudentTCopula", (0.152, 30.607)),
                    (8, 12, {0, 6}, "StudentTCopula", (0.169, 11.887)),
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
                    (0, 1, set(), "GumbelCopula", (1.445, CopulaRotation.R180)),
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
                    (0, 2, {1}, "StudentTCopula", (0.291, 6.732)),
                    (2, 12, {1}, "StudentTCopula", (0.313, 5.621)),
                    (1, 3, {2}, "StudentTCopula", (0.221, 12.199)),
                    (1, 4, {2}, "StudentTCopula", (0.12, 16.893)),
                    (1, 5, {2}, "StudentTCopula", (0.092, 16.746)),
                    (1, 6, {2}, "StudentTCopula", (0.233, 13.367)),
                    (1, 7, {2}, "GumbelCopula", (1.058, CopulaRotation.R180)),
                    (1, 8, {2}, "StudentTCopula", (0.049, 16.261)),
                    (1, 9, {2}, "StudentTCopula", (0.069, 19.857)),
                    (1, 10, {2}, "StudentTCopula", (0.118, 14.929)),
                    (1, 11, {2}, "GaussianCopula", 0.069),
                    (1, 13, {2}, "StudentTCopula", (0.107, 18.037)),
                    (1, 14, {2}, "StudentTCopula", (0.126, 17.192)),
                    (1, 15, {2}, "StudentTCopula", (0.095, 17.232)),
                    (1, 16, {2}, "GaussianCopula", 0.103),
                    (1, 17, {2}, "GaussianCopula", 0.14),
                    (1, 18, {2}, "StudentTCopula", (0.129, 22.231)),
                    (1, 19, {2}, "StudentTCopula", (0.064, 24.333)),
                ],
                [
                    (0, 12, {1, 2}, "StudentTCopula", (0.544, 5.086)),
                    (12, 13, {1, 2}, "StudentTCopula", (0.384, 7.523)),
                    (3, 6, {1, 2}, "StudentTCopula", (0.43, 7.914)),
                    (4, 9, {1, 2}, "StudentTCopula", (0.23, 6.52)),
                    (4, 19, {1, 2}, "StudentTCopula", (0.793, 3.503)),
                    (5, 19, {1, 2}, "StudentTCopula", (0.317, 6.838)),
                    (6, 8, {1, 2}, "StudentTCopula", (0.236, 14.002)),
                    (6, 18, {1, 2}, "StudentTCopula", (0.395, 5.751)),
                    (7, 11, {1, 2}, "StudentTCopula", (0.536, 5.213)),
                    (7, 13, {1, 2}, "StudentTCopula", (0.482, 6.545)),
                    (7, 17, {1, 2}, "StudentTCopula", (0.413, 5.76)),
                    (9, 13, {1, 2}, "StudentTCopula", (0.724, 4.299)),
                    (10, 14, {1, 2}, "StudentTCopula", (0.51, 5.134)),
                    (11, 14, {1, 2}, "StudentTCopula", (0.545, 6.663)),
                    (13, 15, {1, 2}, "StudentTCopula", (0.647, 5.0)),
                    (15, 18, {1, 2}, "StudentTCopula", (0.435, 7.782)),
                    (16, 19, {1, 2}, "StudentTCopula", (0.405, 8.87)),
                ],
                [
                    (0, 13, {1, 2, 12}, "StudentTCopula", (0.16, 17.047)),
                    (12, 7, {1, 2, 13}, "StudentTCopula", (0.165, 17.598)),
                    (3, 18, {1, 2, 6}, "StudentTCopula", (0.112, 11.77)),
                    (9, 19, {1, 2, 4}, "StudentTCopula", (0.079, 12.407)),
                    (4, 13, {9, 2, 1}, "IndependentCopula", None),
                    (4, 5, {1, 2, 19}, "StudentTCopula", (0.071, 32.553)),
                    (4, 16, {1, 2, 19}, "StudentTCopula", (0.133, 29.198)),
                    (8, 18, {1, 2, 6}, "ClaytonCopula", (0.092, CopulaRotation.R0)),
                    (6, 15, {1, 18, 2}, "StudentTCopula", (0.229, 13.35)),
                    (11, 13, {1, 2, 7}, "StudentTCopula", (0.175, 15.855)),
                    (11, 17, {1, 2, 7}, "StudentTCopula", (0.174, 9.348)),
                    (7, 14, {1, 2, 11}, "StudentTCopula", (0.304, 10.27)),
                    (7, 15, {1, 2, 13}, "StudentTCopula", (0.257, 9.967)),
                    (9, 15, {1, 2, 13}, "StudentTCopula", (0.27, 8.773)),
                    (10, 11, {1, 2, 14}, "StudentTCopula", (0.342, 11.803)),
                    (13, 18, {1, 2, 15}, "StudentTCopula", (0.228, 14.153)),
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
                -0.0594782,
                -0.03198484,
                0.00543596,
                -0.06637369,
                -0.01745595,
                -0.01547374,
                -0.06079131,
                0.01851745,
                -0.04212714,
                -0.04366372,
                0.00324127,
                0.0233026,
                -0.03219895,
                0.02528661,
                -0.01482545,
                -0.04618912,
                -0.01757932,
                -0.04627186,
            ],
            [
                -0.2,
                -0.3,
                -0.07883237,
                -0.02098016,
                -0.02253878,
                0.0277289,
                -0.03984467,
                -0.0386899,
                -0.03100295,
                -0.05120378,
                -0.07868818,
                -0.02703185,
                -0.20830192,
                -0.02730654,
                -0.05303455,
                -0.0190775,
                -0.02647807,
                -0.10191758,
                -0.02539445,
                -0.01942394,
            ],
            [
                -0.3,
                -0.4,
                -0.13885623,
                -0.00455346,
                -0.01086378,
                0.01641264,
                -0.04170058,
                -0.12235007,
                -0.09484572,
                -0.09073672,
                -0.23313868,
                -0.19805334,
                -0.17925064,
                -0.08533768,
                -0.1071818,
                -0.04880317,
                -0.0413598,
                0.01337903,
                -0.03353054,
                -0.03036116,
            ],
            [
                -0.4,
                -0.5,
                -0.08510238,
                -0.15718509,
                -0.07510082,
                -0.03555965,
                -0.07200351,
                -0.0767828,
                -0.00535722,
                -0.00551041,
                0.00508226,
                -0.04002202,
                -0.0975833,
                0.02417664,
                -0.01709792,
                -0.02580093,
                -0.12426212,
                -0.12361933,
                -0.03605019,
                -0.0675931,
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
                    0.00716,
                    0.00119,
                    -0.00777,
                    0.02012,
                    -0.00956,
                    -0.03435,
                    0.01573,
                    0.00699,
                    -0.00317,
                    0.00616,
                    0.00516,
                    0.00967,
                    0.01333,
                    -0.00568,
                    -0.00176,
                    -0.00081,
                    -0.03782,
                    0.00726,
                    0.00783,
                    -0.0168,
                ],
                [
                    -0.0182,
                    -0.01555,
                    0.01462,
                    -0.02521,
                    0.00688,
                    0.00247,
                    -0.00686,
                    -0.00233,
                    0.00381,
                    -0.00671,
                    -0.00014,
                    -0.01144,
                    -0.0065,
                    -0.00685,
                    0.00052,
                    -0.00327,
                    0.00449,
                    -0.00025,
                    -0.0048,
                    0.0169,
                ],
            ],
        ),
        (
            4,
            [
                [
                    -0.00006,
                    -0.00348,
                    -0.00128,
                    0.02036,
                    0.01262,
                    0.03853,
                    0.01579,
                    -0.0078,
                    -0.00176,
                    -0.00286,
                    -0.01201,
                    -0.02211,
                    0.01197,
                    0.00013,
                    -0.00008,
                    0.00695,
                    0.01575,
                    -0.00041,
                    -0.00223,
                    0.00545,
                ],
                [
                    0.01406,
                    -0.0031,
                    -0.00361,
                    -0.03153,
                    -0.00304,
                    -0.01556,
                    -0.0069,
                    -0.00582,
                    -0.00514,
                    0.00315,
                    -0.01009,
                    -0.00136,
                    -0.00659,
                    0.00122,
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
