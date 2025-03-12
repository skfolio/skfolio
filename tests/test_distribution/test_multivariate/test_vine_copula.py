import numpy as np
import pytest

from skfolio.distribution import (
    ClaytonCopula,
    CopulaRotation,
    DependenceMethod,
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
            "name": "JohnsonSU",
            "params": {
                "scale_": 0.01506,
                "loc_": 0.00158,
                "a_": 0.02993,
                "b_": 1.19399,
            },
        },
        {
            "name": "JohnsonSU",
            "params": {
                "scale_": 0.03051,
                "loc_": -0.00159,
                "a_": -0.09574,
                "b_": 1.21257,
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
            "name": "JohnsonSU",
            "params": {
                "scale_": 0.01302,
                "loc_": 0.00028,
                "a_": -0.01353,
                "b_": 1.11684,
            },
        },
        {
            "name": "JohnsonSU",
            "params": {
                "scale_": 0.01265,
                "loc_": -0.00108,
                "a_": -0.04587,
                "b_": 0.98439,
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
            "name": "JohnsonSU",
            "params": {
                "scale_": 0.01307,
                "loc_": 0.00121,
                "a_": 0.01063,
                "b_": 1.15195,
            },
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.00709, "loc_": 0.00071, "dof_": 3.23259},
        },
        {
            "name": "JohnsonSU",
            "params": {
                "scale_": 0.01127,
                "loc_": -0.00031,
                "a_": -0.05647,
                "b_": 1.17291,
            },
        },
        {
            "name": "StudentT",
            "params": {"scale_": 0.0071, "loc_": 0.0006, "dof_": 3.09206},
        },
        {
            "name": "JohnsonSU",
            "params": {
                "scale_": 0.04371,
                "loc_": -0.00742,
                "a_": -0.20311,
                "b_": 1.47876,
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
            "name": "JohnsonSU",
            "params": {
                "scale_": 0.01273,
                "loc_": -0.0002,
                "a_": -0.0341,
                "b_": 1.11558,
            },
        },
    ]


@pytest.fixture
def expected_trees():
    return [
        [
            (0, 12, set(), "StudentTCopula", (0.658, 2.677)),
            (1, 12, set(), "GumbelCopula", (1.471, CopulaRotation.R180)),
            (2, 8, set(), "StudentTCopula", (0.898, 3.068)),
            (3, 6, set(), "StudentTCopula", (0.553, 4.8)),
            (4, 8, set(), "StudentTCopula", (0.546, 4.011)),
            (4, 19, set(), "StudentTCopula", (0.853, 3.463)),
            (5, 8, set(), "StudentTCopula", (0.557, 3.463)),
            (6, 8, set(), "StudentTCopula", (0.457, 3.714)),
            (6, 12, set(), "StudentTCopula", (0.509, 3.833)),
            (6, 18, set(), "StudentTCopula", (0.454, 3.78)),
            (7, 11, set(), "StudentTCopula", (0.577, 4.002)),
            (7, 15, set(), "StudentTCopula", (0.517, 3.898)),
            (7, 17, set(), "StudentTCopula", (0.48, 3.838)),
            (9, 13, set(), "StudentTCopula", (0.745, 3.431)),
            (10, 14, set(), "StudentTCopula", (0.553, 3.791)),
            (11, 14, set(), "StudentTCopula", (0.598, 4.852)),
            (13, 15, set(), "StudentTCopula", (0.675, 3.75)),
            (15, 18, set(), "StudentTCopula", (0.472, 4.93)),
            (16, 19, set(), "StudentTCopula", (0.495, 5.371)),
        ],
        [
            (0, 1, {12}, "StudentTCopula", (0.215, 12.542)),
            (0, 6, {12}, "StudentTCopula", (0.181, 18.598)),
            (2, 5, {8}, "GaussianCopula", 0.109),
            (3, 8, {6}, "StudentTCopula", (0.26, 11.079)),
            (8, 19, {4}, "StudentTCopula", (0.196, 9.264)),
            (4, 5, {8}, "StudentTCopula", (0.27, 11.093)),
            (4, 6, {8}, "StudentTCopula", (0.113, 7.479)),
            (4, 16, {19}, "StudentTCopula", (0.153, 16.45)),
            (8, 12, {6}, "StudentTCopula", (0.275, 4.443)),
            (12, 18, {6}, "StudentTCopula", (0.165, 8.678)),
            (6, 15, {18}, "StudentTCopula", (0.264, 6.968)),
            (11, 17, {7}, "StudentTCopula", (0.215, 7.467)),
            (7, 14, {11}, "StudentTCopula", (0.325, 7.324)),
            (15, 17, {7}, "StudentTCopula", (0.176, 11.657)),
            (7, 13, {15}, "StudentTCopula", (0.276, 8.221)),
            (9, 15, {13}, "StudentTCopula", (0.286, 7.061)),
            (10, 11, {14}, "StudentTCopula", (0.35, 10.278)),
            (13, 18, {15}, "StudentTCopula", (0.238, 10.944)),
        ],
        [
            (1, 6, {0, 12}, "GaussianCopula", 0.085),
            (0, 8, {12, 6}, "GumbelCopula", (1.115, CopulaRotation.R180)),
            (2, 4, {8, 5}, "StudentTCopula", (0.067, 13.277)),
            (3, 4, {8, 6}, "StudentTCopula", (0.083, 21.881)),
            (19, 5, {8, 4}, "StudentTCopula", (0.138, 15.661)),
            (8, 16, {19, 4}, "IndependentCopula", None),
            (5, 6, {8, 4}, "StudentTCopula", (0.085, 12.993)),
            (4, 12, {8, 6}, "StudentTCopula", (0.118, 12.096)),
            (8, 18, {12, 6}, "StudentTCopula", (0.065, 17.316)),
            (12, 15, {18, 6}, "StudentTCopula", (0.231, 8.472)),
            (6, 13, {18, 15}, "StudentTCopula", (0.187, 8.708)),
            (17, 14, {11, 7}, "StudentTCopula", (0.183, 12.224)),
            (11, 15, {17, 7}, "StudentTCopula", (0.141, 13.679)),
            (7, 10, {11, 14}, "StudentTCopula", (0.278, 18.24)),
            (17, 13, {7, 15}, "GumbelCopula", (1.079, CopulaRotation.R180)),
            (7, 9, {13, 15}, "StudentTCopula", (0.107, 10.433)),
            (7, 18, {13, 15}, "StudentTCopula", (0.149, 14.702)),
        ],
        [
            (1, 8, {0, 12, 6}, "GaussianCopula", 0.069),
            (0, 18, {8, 12, 6}, "GaussianCopula", 0.057),
            (2, 6, {8, 4, 5}, "IndependentCopula", None),
            (3, 5, {8, 4, 6}, "GaussianCopula", 0.07),
            (3, 12, {8, 4, 6}, "ClaytonCopula", (0.069, CopulaRotation.R180)),
            (5, 16, {8, 19, 4}, "StudentTCopula", (0.066, 15.992)),
            (19, 6, {8, 4, 5}, "IndependentCopula", None),
            (4, 18, {8, 12, 6}, "StudentTCopula", (0.067, 11.861)),
            (8, 15, {18, 12, 6}, "StudentTCopula", (0.07, 10.413)),
            (12, 13, {18, 6, 15}, "StudentTCopula", (0.169, 12.941)),
            (6, 7, {18, 13, 15}, "StudentTCopula", (0.157, 13.95)),
            (14, 15, {17, 11, 7}, "StudentTCopula", (0.081, 18.335)),
            (17, 10, {11, 14, 7}, "StudentTCopula", (0.132, 24.724)),
            (11, 13, {17, 7, 15}, "GumbelCopula", (1.055, CopulaRotation.R180)),
            (17, 9, {7, 13, 15}, "StudentTCopula", (0.059, 13.694)),
            (17, 18, {7, 13, 15}, "StudentTCopula", (0.136, 16.533)),
        ],
        [
            (1, 18, {0, 8, 12, 6}, "IndependentCopula", None),
            (0, 4, {8, 18, 12, 6}, "IndependentCopula", None),
            (2, 3, {8, 4, 5, 6}, "StudentTCopula", (0.075, 18.971)),
            (2, 19, {8, 4, 5, 6}, "IndependentCopula", None),
            (5, 12, {8, 3, 4, 6}, "StudentTCopula", (0.049, 23.594)),
            (3, 18, {8, 4, 12, 6}, "StudentTCopula", (0.108, 13.488)),
            (16, 6, {8, 19, 4, 5}, "StudentTCopula", (-0.0434, 16.57)),
            (4, 15, {8, 18, 12, 6}, "StudentTCopula", (0.057, 11.517)),
            (8, 13, {18, 12, 6, 15}, "IndependentCopula", None),
            (12, 7, {18, 13, 6, 15}, "StudentTCopula", (0.138, 22.121)),
            (6, 17, {18, 15, 13, 7}, "StudentTCopula", (0.231, 20.335)),
            (15, 10, {17, 11, 14, 7}, "IndependentCopula", None),
            (14, 13, {17, 11, 15, 7}, "GaussianCopula", 0.043),
            (11, 9, {17, 15, 13, 7}, "GumbelCopula", (1.032, CopulaRotation.R180)),
            (9, 18, {17, 15, 13, 7}, "IndependentCopula", None),
        ],
        [
            (1, 4, {0, 6, 8, 12, 18}, "GaussianCopula", 0.074),
            (0, 3, {4, 6, 8, 12, 18}, "GaussianCopula", 0.137),
            (3, 19, {2, 4, 5, 6, 8}, "IndependentCopula", None),
            (2, 12, {3, 4, 5, 6, 8}, "IndependentCopula", None),
            (2, 16, {4, 5, 6, 8, 19}, "GaussianCopula", 0.097),
            (5, 18, {3, 4, 6, 8, 12}, "StudentTCopula", (0.053, 28.25)),
            (3, 15, {4, 6, 8, 12, 18}, "StudentTCopula", (-0.059, 24.602)),
            (4, 13, {6, 8, 12, 15, 18}, "StudentTCopula", (0.06, 23.614)),
            (8, 7, {6, 12, 13, 15, 18}, "StudentTCopula", (0.158, 16.712)),
            (12, 17, {6, 7, 13, 15, 18}, "GaussianCopula", 0.182),
            (6, 9, {7, 13, 15, 17, 18}, "GumbelCopula", (1.03, CopulaRotation.R0)),
            (10, 13, {7, 11, 14, 15, 17}, "IndependentCopula", None),
            (14, 9, {7, 11, 13, 15, 17}, "IndependentCopula", None),
            (11, 18, {7, 9, 13, 15, 17}, "IndependentCopula", None),
        ],
        [
            (1, 3, {0, 4, 6, 8, 12, 18}, "ClaytonCopula", (0.135, CopulaRotation.R0)),
            (0, 5, {3, 4, 6, 8, 12, 18}, "StudentTCopula", (0.093, 28.737)),
            (19, 12, {2, 3, 4, 5, 6, 8}, "IndependentCopula", None),
            (3, 16, {2, 4, 5, 6, 8, 19}, "GaussianCopula", 0.09),
            (2, 18, {3, 4, 5, 6, 8, 12}, "IndependentCopula", None),
            (5, 15, {3, 4, 6, 8, 12, 18}, "IndependentCopula", None),
            (3, 13, {4, 6, 8, 12, 15, 18}, "GaussianCopula", -0.057),
            (4, 7, {6, 8, 12, 13, 15, 18}, "GumbelCopula", (1.053, CopulaRotation.R0)),
            (8, 17, {6, 7, 12, 13, 15, 18}, "StudentTCopula", (0.217, 10.815)),
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
            (5, 13, {3, 4, 6, 8, 12, 15, 18}, "StudentTCopula", (0.044, 31.282)),
            (3, 7, {4, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (4, 17, {6, 7, 8, 12, 13, 15, 18}, "StudentTCopula", (0.052, 31.787)),
            (8, 9, {6, 7, 12, 13, 15, 17, 18}, "StudentTCopula", (0.153, 10.724)),
            (12, 11, {6, 7, 9, 13, 15, 17, 18}, "StudentTCopula", (0.093, 26.646)),
            (6, 14, {7, 9, 11, 13, 15, 17, 18}, "GaussianCopula", 0.087),
            (10, 18, {7, 9, 11, 13, 14, 15, 17}, "IndependentCopula", None),
        ],
        [
            (1, 15, {0, 3, 4, 5, 6, 8, 12, 18}, "GaussianCopula", -0.07),
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
            (4, 9, {6, 7, 8, 12, 13, 15, 17, 18}, "StudentTCopula", (0.149, 24.452)),
            (8, 11, {6, 7, 9, 12, 13, 15, 17, 18}, "StudentTCopula", (0.099, 32.757)),
            (12, 14, {6, 7, 9, 11, 13, 15, 17, 18}, "StudentTCopula", (0.052, 22.87)),
            (6, 10, {7, 9, 11, 13, 14, 15, 17, 18}, "IndependentCopula", None),
        ],
        [
            (1, 13, {0, 3, 4, 5, 6, 8, 12, 15, 18}, "GaussianCopula", -0.069),
            (0, 2, {3, 4, 5, 6, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (
                16,
                15,
                {2, 3, 4, 5, 6, 8, 12, 18, 19},
                "StudentTCopula",
                (-0.055, 18.329),
            ),
            (
                19,
                13,
                {2, 3, 4, 5, 6, 8, 12, 15, 18},
                "GumbelCopula",
                (1.031, CopulaRotation.R180),
            ),
            (
                2,
                7,
                {3, 4, 5, 6, 8, 12, 13, 15, 18},
                "ClaytonCopula",
                (0.057, CopulaRotation.R270),
            ),
            (5, 17, {3, 4, 6, 7, 8, 12, 13, 15, 18}, "IndependentCopula", None),
            (3, 9, {4, 6, 7, 8, 12, 13, 15, 17, 18}, "GaussianCopula", -0.05),
            (
                4,
                11,
                {6, 7, 8, 9, 12, 13, 15, 17, 18},
                "StudentTCopula",
                (0.093, 25.269),
            ),
            (
                8,
                14,
                {6, 7, 9, 11, 12, 13, 15, 17, 18},
                "StudentTCopula",
                (0.082, 15.969),
            ),
            (
                12,
                10,
                {6, 7, 9, 11, 13, 14, 15, 17, 18},
                "StudentTCopula",
                (0.108, 31.967),
            ),
        ],
        [
            (
                1,
                2,
                {0, 3, 4, 5, 6, 8, 12, 13, 15, 18},
                "StudentTCopula",
                (0.084, 24.929),
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
                (0.072, 29.067),
            ),
            (3, 11, {4, 6, 7, 8, 9, 12, 13, 15, 17, 18}, "IndependentCopula", None),
            (4, 14, {6, 7, 8, 9, 11, 12, 13, 15, 17, 18}, "IndependentCopula", None),
            (
                8,
                10,
                {6, 7, 9, 11, 12, 13, 14, 15, 17, 18},
                "StudentTCopula",
                (-0.047, 41.265),
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
            (16, 7, {2, 3, 4, 5, 6, 8, 12, 13, 15, 18, 19}, "GaussianCopula", -0.08),
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
            (1, 16, {0, 2, 3, 4, 5, 6, 8, 12, 13, 15, 18, 19}, "GaussianCopula", 0.073),
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
                (-0.044, 25.049),
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
                (0.054, 20.005),
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
                "GaussianCopula",
                0.044,
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
    # s = time.time()
    model = VineCopula(n_jobs=-1, max_depth=100)
    model.fit(X)
    # e = time.time()
    # assert (e - s) < 20 # For local sanity check
    # 10s
    _check_vine_output(model, expected_marginals, expected_trees)
    sample = model.sample(n_samples=1000)
    assert sample.shape == (1000, 20)
    with pytest.warns(UserWarning, match="^When performing conditional sampling"):
        sample = model.sample(
            n_samples=1000,
            conditioning={
                0: -0.4,
                1: (None, -0.5),
                2: np.full(1000, -0.3),
            },
        )
        assert sample.shape == (1000, 20)

    _ = model.fitted_repr

    _ = model.plot_scatter_matrix(X)
    with pytest.warns(UserWarning, match="^When performing conditional sampling"):
        _ = model.plot_scatter_matrix(
            conditioning={
                0: -0.4,
                1: (None, -0.5),
                2: np.full(1000, -0.3),
            },
        )
        _ = model.plot_marginal_distributions(
            X,
            n_samples=len(X),
            conditioning={
                0: -0.4,
                1: (None, -0.5),
                2: np.full(len(X), -0.3),
            },
        )
    model.display_vine()


def test_log_transform(X):
    model = VineCopula(
        n_jobs=-1,
        max_depth=3,
        marginal_candidates=[Gaussian()],
        copula_candidates=[GaussianCopula()],
        central_assets=[0],
        log_transform=True,
    )
    model.fit(X)
    sample = model.sample(n_samples=1000, conditioning={0: np.full(1000, -0.8)})
    assert np.all(sample >= -1)


@pytest.mark.parametrize(
    "max_depth,central_assets,expected_trees",
    [
        (
            3,
            ["AAPL"],
            [
                [
                    (0, 1, set(), "StudentTCopula", (0.465, 5.146)),
                    (0, 2, set(), "StudentTCopula", (0.373, 3.643)),
                    (0, 3, set(), "StudentTCopula", (0.39, 6.974)),
                    (0, 4, set(), "StudentTCopula", (0.295, 4.047)),
                    (0, 5, set(), "StudentTCopula", (0.326, 3.552)),
                    (0, 6, set(), "StudentTCopula", (0.435, 3.553)),
                    (0, 7, set(), "StudentTCopula", (0.306, 5.617)),
                    (0, 8, set(), "StudentTCopula", (0.413, 3.366)),
                    (0, 9, set(), "StudentTCopula", (0.307, 3.883)),
                    (0, 10, set(), "StudentTCopula", (0.286, 5.318)),
                    (0, 11, set(), "StudentTCopula", (0.267, 5.532)),
                    (0, 12, set(), "StudentTCopula", (0.658, 2.677)),
                    (0, 13, set(), "StudentTCopula", (0.369, 3.955)),
                    (0, 14, set(), "StudentTCopula", (0.313, 4.428)),
                    (0, 15, set(), "StudentTCopula", (0.311, 4.36)),
                    (0, 16, set(), "StudentTCopula", (0.184, 6.461)),
                    (0, 17, set(), "StudentTCopula", (0.369, 5.011)),
                    (0, 18, set(), "StudentTCopula", (0.304, 5.469)),
                    (0, 19, set(), "StudentTCopula", (0.3, 3.996)),
                ],
                [
                    (1, 12, {0}, "StudentTCopula", (0.273, 9.444)),
                    (2, 8, {0}, "StudentTCopula", (0.875, 3.406)),
                    (3, 6, {0}, "StudentTCopula", (0.464, 7.25)),
                    (4, 8, {0}, "StudentTCopula", (0.481, 5.682)),
                    (4, 19, {0}, "StudentTCopula", (0.837, 4.016)),
                    (5, 8, {0}, "StudentTCopula", (0.484, 5.18)),
                    (6, 8, {0}, "StudentTCopula", (0.343, 6.512)),
                    (6, 12, {0}, "StudentTCopula", (0.327, 10.994)),
                    (6, 18, {0}, "StudentTCopula", (0.373, 5.078)),
                    (7, 11, {0}, "StudentTCopula", (0.547, 5.173)),
                    (7, 15, {0}, "StudentTCopula", (0.461, 5.229)),
                    (7, 17, {0}, "StudentTCopula", (0.416, 5.324)),
                    (9, 13, {0}, "StudentTCopula", (0.711, 4.173)),
                    (10, 14, {0}, "StudentTCopula", (0.507, 5.076)),
                    (11, 14, {0}, "StudentTCopula", (0.56, 6.344)),
                    (13, 15, {0}, "StudentTCopula", (0.629, 4.543)),
                    (15, 18, {0}, "StudentTCopula", (0.415, 7.817)),
                    (16, 19, {0}, "StudentTCopula", (0.468, 6.769)),
                ],
                [
                    (1, 6, {0, 12}, "StudentTCopula", (0.092, 31.6)),
                    (2, 5, {8, 0}, "GaussianCopula", 0.114),
                    (3, 8, {0, 6}, "StudentTCopula", (0.222, 13.873)),
                    (3, 18, {0, 6}, "StudentTCopula", (0.112, 11.058)),
                    (8, 19, {0, 4}, "StudentTCopula", (0.178, 13.153)),
                    (4, 5, {8, 0}, "StudentTCopula", (0.262, 16.03)),
                    (4, 6, {8, 0}, "StudentTCopula", (0.084, 9.346)),
                    (4, 16, {0, 19}, "StudentTCopula", (0.151, 32.266)),
                    (8, 12, {0, 6}, "StudentTCopula", (0.171, 12.214)),
                    (6, 15, {0, 18}, "StudentTCopula", (0.204, 9.696)),
                    (11, 15, {0, 7}, "StudentTCopula", (0.157, 12.131)),
                    (11, 17, {0, 7}, "StudentTCopula", (0.189, 8.638)),
                    (7, 14, {0, 11}, "StudentTCopula", (0.296, 9.464)),
                    (7, 13, {0, 15}, "StudentTCopula", (0.244, 9.197)),
                    (9, 15, {0, 13}, "StudentTCopula", (0.279, 7.945)),
                    (10, 11, {0, 14}, "StudentTCopula", (0.34, 10.991)),
                    (13, 18, {0, 15}, "StudentTCopula", (0.201, 14.0)),
                ],
            ],
        ),
        (
            4,
            [1, 2],
            [
                [
                    (0, 1, set(), "StudentTCopula", (0.465, 5.146)),
                    (1, 2, set(), "GumbelCopula", (1.255, CopulaRotation.R180)),
                    (1, 12, set(), "GumbelCopula", (1.471, CopulaRotation.R180)),
                    (2, 3, set(), "StudentTCopula", (0.417, 5.034)),
                    (2, 4, set(), "StudentTCopula", (0.519, 4.77)),
                    (2, 5, set(), "StudentTCopula", (0.538, 4.129)),
                    (2, 6, set(), "StudentTCopula", (0.409, 4.362)),
                    (2, 7, set(), "StudentTCopula", (0.293, 4.18)),
                    (2, 8, set(), "StudentTCopula", (0.898, 3.068)),
                    (2, 9, set(), "StudentTCopula", (0.304, 3.423)),
                    (2, 10, set(), "StudentTCopula", (0.241, 4.888)),
                    (2, 11, set(), "StudentTCopula", (0.294, 5.032)),
                    (2, 13, set(), "StudentTCopula", (0.224, 3.986)),
                    (2, 14, set(), "StudentTCopula", (0.326, 3.82)),
                    (2, 15, set(), "StudentTCopula", (0.23, 3.408)),
                    (2, 16, set(), "StudentTCopula", (0.332, 7.526)),
                    (2, 17, set(), "StudentTCopula", (0.401, 4.074)),
                    (2, 18, set(), "StudentTCopula", (0.244, 4.603)),
                    (2, 19, set(), "StudentTCopula", (0.52, 4.318)),
                ],
                [
                    (0, 2, {1}, "StudentTCopula", (0.29, 6.836)),
                    (2, 12, {1}, "StudentTCopula", (0.313, 5.706)),
                    (1, 3, {2}, "StudentTCopula", (0.222, 12.157)),
                    (1, 4, {2}, "StudentTCopula", (0.121, 17.497)),
                    (1, 5, {2}, "StudentTCopula", (0.093, 16.457)),
                    (1, 6, {2}, "StudentTCopula", (0.234, 13.394)),
                    (1, 7, {2}, "GumbelCopula", (1.058, CopulaRotation.R180)),
                    (1, 8, {2}, "StudentTCopula", (0.049, 16.987)),
                    (1, 9, {2}, "StudentTCopula", (0.069, 20.653)),
                    (1, 10, {2}, "StudentTCopula", (0.118, 15.685)),
                    (1, 11, {2}, "GaussianCopula", 0.069),
                    (1, 13, {2}, "StudentTCopula", (0.108, 18.501)),
                    (1, 14, {2}, "StudentTCopula", (0.126, 17.163)),
                    (1, 15, {2}, "StudentTCopula", (0.096, 17.402)),
                    (1, 16, {2}, "GaussianCopula", 0.103),
                    (1, 17, {2}, "GaussianCopula", 0.141),
                    (1, 18, {2}, "StudentTCopula", (0.129, 22.125)),
                    (1, 19, {2}, "StudentTCopula", (0.064, 25.891)),
                ],
                [
                    (0, 12, {1, 2}, "StudentTCopula", (0.539, 5.335)),
                    (12, 13, {1, 2}, "StudentTCopula", (0.383, 7.558)),
                    (3, 6, {1, 2}, "StudentTCopula", (0.43, 7.936)),
                    (4, 9, {1, 2}, "StudentTCopula", (0.23, 6.625)),
                    (4, 19, {1, 2}, "StudentTCopula", (0.792, 3.526)),
                    (5, 19, {1, 2}, "StudentTCopula", (0.317, 6.716)),
                    (6, 8, {1, 2}, "StudentTCopula", (0.236, 13.919)),
                    (6, 18, {1, 2}, "StudentTCopula", (0.395, 5.773)),
                    (7, 11, {1, 2}, "StudentTCopula", (0.536, 5.206)),
                    (7, 13, {1, 2}, "StudentTCopula", (0.482, 6.539)),
                    (7, 17, {1, 2}, "StudentTCopula", (0.413, 5.754)),
                    (9, 13, {1, 2}, "StudentTCopula", (0.724, 4.296)),
                    (10, 14, {1, 2}, "StudentTCopula", (0.51, 5.123)),
                    (11, 14, {1, 2}, "StudentTCopula", (0.545, 6.659)),
                    (13, 15, {1, 2}, "StudentTCopula", (0.647, 4.997)),
                    (15, 18, {1, 2}, "StudentTCopula", (0.435, 7.798)),
                    (16, 19, {1, 2}, "StudentTCopula", (0.405, 8.888)),
                ],
                [
                    (0, 13, {1, 2, 12}, "StudentTCopula", (0.17, 18.522)),
                    (12, 7, {1, 2, 13}, "StudentTCopula", (0.166, 18.186)),
                    (3, 18, {1, 2, 6}, "StudentTCopula", (0.113, 11.777)),
                    (9, 19, {1, 2, 4}, "StudentTCopula", (0.079, 12.271)),
                    (4, 13, {9, 2, 1}, "IndependentCopula", None),
                    (4, 5, {1, 2, 19}, "StudentTCopula", (0.072, 34.646)),
                    (4, 16, {1, 2, 19}, "StudentTCopula", (0.132, 31.219)),
                    (8, 18, {1, 2, 6}, "ClaytonCopula", (0.092, CopulaRotation.R0)),
                    (6, 15, {1, 18, 2}, "StudentTCopula", (0.229, 13.278)),
                    (11, 13, {1, 2, 7}, "StudentTCopula", (0.175, 15.813)),
                    (11, 17, {1, 2, 7}, "StudentTCopula", (0.174, 9.347)),
                    (7, 14, {1, 2, 11}, "StudentTCopula", (0.303, 10.202)),
                    (7, 15, {1, 2, 13}, "StudentTCopula", (0.257, 9.96)),
                    (9, 15, {1, 2, 13}, "StudentTCopula", (0.27, 8.787)),
                    (10, 11, {1, 2, 14}, "StudentTCopula", (0.343, 11.826)),
                    (13, 18, {1, 2, 15}, "StudentTCopula", (0.228, 14.144)),
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
    model = VineCopula(n_jobs=-1, max_depth=3, central_assets=["AMD", "BAC"])
    model.fit(X)
    sample1 = model.sample(
        n_samples=4,
        random_state=42,
        conditioning={
            1: [-0.1, -0.2, -0.3, -0.4],
            2: [-0.2, -0.3, -0.4, -0.5],
        },
    )
    sample2 = model.sample(
        n_samples=4,
        random_state=42,
        conditioning={
            "AMD": [-0.1, -0.2, -0.3, -0.4],
            "BAC": [-0.2, -0.3, -0.4, -0.5],
        },
    )
    np.testing.assert_array_almost_equal(sample1, sample2)
    np.testing.assert_array_almost_equal(
        sample1,
        [
            [
                -0.12343,
                -0.1,
                -0.2,
                -0.10225,
                -0.17206,
                -0.22719,
                -0.04168,
                -0.02158,
                -0.21725,
                -0.02352,
                0.02392,
                -0.06747,
                -0.16393,
                -0.03277,
                -0.00813,
                0.03742,
                0.04616,
                -0.10275,
                -0.04356,
                -0.13262,
            ],
            [
                -0.03673,
                -0.2,
                -0.3,
                0.01994,
                -0.11496,
                0.01458,
                -0.18415,
                -0.1191,
                -0.42162,
                -0.04984,
                -0.01337,
                -0.10517,
                -0.10518,
                -0.0407,
                -0.04131,
                -0.08433,
                -0.24517,
                -0.25315,
                -0.09034,
                -0.11494,
            ],
            [
                -0.05852,
                -0.3,
                -0.4,
                -0.01557,
                -0.32027,
                -0.16369,
                -0.32234,
                -0.08957,
                -0.19844,
                -0.08869,
                0.00006,
                -0.06939,
                -0.19984,
                -0.09353,
                0.0235,
                -0.23232,
                -0.14847,
                -0.27818,
                -0.30915,
                -0.2587,
            ],
            [
                -0.2439,
                -0.4,
                -0.5,
                -0.04271,
                -0.21955,
                0.16408,
                -0.07562,
                -0.14834,
                -0.66828,
                -0.25275,
                -0.38596,
                -0.15458,
                -0.23565,
                -0.02741,
                -0.17342,
                0.10351,
                -0.12848,
                -0.0663,
                -0.27624,
                -0.22994,
            ],
        ],
        5,
    )


def test_vine_copula_conditional_sampling_without_priority(X):
    model = VineCopula(
        n_jobs=-1,
        marginal_candidates=[Gaussian()],
        copula_candidates=[GaussianCopula()],
    )
    model.fit(X)

    for i in range(20):
        with pytest.warns(UserWarning, match="^When performing conditional sampling"):
            sample = model.sample(
                n_samples=1000, conditioning={i: -np.ones(1000) * 0.5}
            )
            assert sample.shape == (1000, 20)


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
                    0.00733,
                    0.00137,
                    -0.00812,
                    0.02013,
                    -0.00941,
                    -0.03391,
                    0.01573,
                    0.007,
                    -0.00317,
                    0.00614,
                    0.00517,
                    0.00968,
                    0.01321,
                    -0.00567,
                    -0.00181,
                    -0.00081,
                    -0.03806,
                    0.00726,
                    0.00783,
                    -0.01661,
                ],
                [
                    -0.018,
                    -0.01518,
                    0.01468,
                    -0.02521,
                    0.00675,
                    0.00256,
                    -0.00686,
                    -0.00233,
                    0.00381,
                    -0.00671,
                    -0.00015,
                    -0.01145,
                    -0.00642,
                    -0.00685,
                    0.00051,
                    -0.00327,
                    0.00463,
                    -0.00025,
                    -0.0048,
                    0.01685,
                ],
            ],
        ),
        (
            4,
            [
                [
                    -0.00006,
                    -0.00354,
                    -0.00128,
                    0.02036,
                    0.01268,
                    0.03793,
                    0.01579,
                    -0.0078,
                    -0.00176,
                    -0.00286,
                    -0.01201,
                    -0.02211,
                    0.01182,
                    0.00013,
                    -0.00012,
                    0.00695,
                    0.01602,
                    -0.00041,
                    -0.00223,
                    0.00558,
                ],
                [
                    0.01413,
                    -0.00359,
                    -0.00361,
                    -0.03154,
                    -0.00294,
                    -0.01549,
                    -0.0069,
                    -0.00582,
                    -0.00514,
                    0.00315,
                    -0.01009,
                    -0.00136,
                    -0.00651,
                    0.00122,
                    -0.01383,
                    -0.00365,
                    -0.03678,
                    -0.00002,
                    0.00295,
                    0.0052,
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


@pytest.mark.filterwarnings("ignore: When performing conditional sampling")
@pytest.mark.parametrize(
    "conditioning,log_transform,expected",
    [
        (
            {
                "BAC": -0.5,
                "UNH": (-0.6, -0.3),
                "WMT": (None, 0.4),
                "XOM": (-np.inf, 0),
                "AMD": (-0.4, None),
                "AAPL": (0.3, np.inf),
                "JPM": [-0.4, 0.5, -0.8],
            },
            False,
            [
                [
                    3.00000000e-01,
                    2.23361833e-02,
                    -5.00000000e-01,
                    -8.37385792e-02,
                    -5.62195153e-02,
                    -1.00365557e-01,
                    -5.35035024e-02,
                    -3.00720824e-02,
                    -4.00000000e-01,
                    -3.95890504e-02,
                    -5.41594024e-02,
                    -3.65940867e-02,
                    -5.88657922e-02,
                    -3.86054663e-02,
                    -3.13603278e-02,
                    -3.46842730e-02,
                    -5.60067396e-02,
                    -3.00000000e-01,
                    3.78465419e-03,
                    -3.30471145e-02,
                ],
                [
                    3.00000000e-01,
                    -7.42699113e-02,
                    -5.00000000e-01,
                    6.23717984e-02,
                    8.07557214e-02,
                    1.09366667e-01,
                    4.59916512e-02,
                    -6.14214345e-03,
                    5.00000000e-01,
                    7.27075169e-03,
                    -2.78294527e-02,
                    -2.59763336e-02,
                    2.61266737e-02,
                    -7.24535223e-04,
                    -1.66803455e-02,
                    1.10497426e-02,
                    -2.62678958e-02,
                    -3.00000000e-01,
                    -1.31188006e-02,
                    -2.91580770e-03,
                ],
                [
                    3.00000000e-01,
                    7.20163400e-02,
                    -5.00000000e-01,
                    -8.71065125e-02,
                    -7.18393166e-02,
                    -7.97826907e-02,
                    -4.13589895e-02,
                    -1.89667497e-02,
                    -8.00000000e-01,
                    -2.41698410e-02,
                    -6.83587073e-03,
                    -6.44280560e-03,
                    -7.21832434e-02,
                    -3.25372947e-02,
                    -1.55405794e-02,
                    -2.06755591e-02,
                    -4.22767854e-03,
                    -3.00000000e-01,
                    -1.31201519e-02,
                    -9.06935089e-03,
                ],
            ],
        ),
        (
            {
                "BAC": -0.5,
                "UNH": (-0.6, -0.3),
                "WMT": (None, 0.4),
                "XOM": (-np.inf, 0),
                "AMD": (-0.4, None),
                "AAPL": (0.3, np.inf),
                "JPM": [-0.4, 0.5, -0.8],
            },
            True,
            [
                [
                    3.00000000e-01,
                    2.15627296e-02,
                    -5.00000000e-01,
                    -8.12393281e-02,
                    -5.51539902e-02,
                    -9.56352985e-02,
                    -5.26743763e-02,
                    -2.97671547e-02,
                    -4.00000000e-01,
                    -3.90878963e-02,
                    -5.24846743e-02,
                    -3.59715709e-02,
                    -5.72938959e-02,
                    -3.80583914e-02,
                    -3.09120548e-02,
                    -3.42077491e-02,
                    -5.44247410e-02,
                    -3.00000000e-01,
                    3.70081974e-03,
                    -3.26152382e-02,
                ],
                [
                    3.00000000e-01,
                    -7.10971050e-02,
                    -5.00000000e-01,
                    6.45757123e-02,
                    8.44288191e-02,
                    1.15321191e-01,
                    4.73676172e-02,
                    -6.18180754e-03,
                    5.00000000e-01,
                    7.26191541e-03,
                    -2.74669282e-02,
                    -2.59898545e-02,
                    2.63173766e-02,
                    -7.84705432e-04,
                    -1.66880454e-02,
                    1.10442777e-02,
                    -2.66619648e-02,
                    -3.00000000e-01,
                    -1.31186773e-02,
                    -2.93099323e-03,
                ],
                [
                    3.00000000e-01,
                    7.27562499e-02,
                    -5.00000000e-01,
                    -8.43497177e-02,
                    -6.98531538e-02,
                    -7.68862505e-02,
                    -4.09943644e-02,
                    -1.88987464e-02,
                    -8.00000000e-01,
                    -2.40980253e-02,
                    -6.95830048e-03,
                    -6.65538379e-03,
                    -6.97614615e-02,
                    -3.21778957e-02,
                    -1.55518121e-02,
                    -2.05701088e-02,
                    -4.43187307e-03,
                    -3.00000000e-01,
                    -1.31200105e-02,
                    -9.07857418e-03,
                ],
            ],
        ),
    ],
)
def test_vine_conditional_sample(X, conditioning, log_transform, expected):
    model = VineCopula(
        marginal_candidates=[Gaussian()],
        copula_candidates=[GaussianCopula(), ClaytonCopula()],
        log_transform=log_transform,
        max_depth=5,
    )
    model.fit(X)
    sample = model.sample(n_samples=3, random_state=42, conditioning=conditioning)
    np.testing.assert_array_almost_equal(sample, expected, 5)


@pytest.mark.parametrize("dependence_method", list(DependenceMethod))
def test_vine_dependence_method(X, dependence_method):
    model = VineCopula(
        marginal_candidates=[Gaussian()],
        copula_candidates=[GaussianCopula(), ClaytonCopula()],
        dependence_method=dependence_method,
        n_jobs=-1,
    )
    model.fit(X)


def test_vine_fit_raise():
    with pytest.raises(ValueError, match="The number of assets must be higher than 2"):
        X = np.array([[0.1, 0.2], [0.2, 0.25], [0.3, 0.35]])
        model = VineCopula(
            marginal_candidates=[Gaussian()],
            copula_candidates=[GaussianCopula()],
            n_jobs=-1,
        )
        model.fit(X)

    with pytest.raises(ValueError, match="`max_depth` must be higher than 1"):
        X = np.array([[0.1, 0.2, 0.5], [0.2, 0.25, 0.6], [0.3, 0.35, 0.7]])
        model = VineCopula(
            marginal_candidates=[Gaussian()],
            copula_candidates=[GaussianCopula()],
            n_jobs=-1,
            max_depth=1,
        )
        model.fit(X)

    with pytest.raises(ValueError, match="X must be in the interval"):
        X = np.array([[0.1, 0.2, 0.5], [0.2, 0.25, 0.6], [0.3, 0.35, 1.7]])
        model = VineCopula(
            marginal_candidates=[Gaussian()],
            copula_candidates=[GaussianCopula()],
            n_jobs=-1,
            fit_marginals=False,
        )
        model.fit(X)


@pytest.mark.filterwarnings("ignore: When performing conditional sampling")
def test_vine_sample_raise(X):
    model = VineCopula(
        marginal_candidates=[Gaussian()],
        copula_candidates=[GaussianCopula()],
    )
    model.fit(X)

    with pytest.raises(
        ValueError, match="`conditioning` must be provided for strictly"
    ):
        _ = model.sample(conditioning={name: [0.5] for name in X.columns})
    with pytest.raises(ValueError, match="When an array is provided"):
        _ = model.sample(conditioning={"AAPL": [[0.5], [0.5]]})


def test_vine_plot_raise(X):
    model = VineCopula(
        marginal_candidates=[Gaussian()],
        copula_candidates=[GaussianCopula()],
    )
    model.fit(X)

    with pytest.raises(ValueError, match="X should be an 2D array"):
        X = np.array([[[0.1, 0.2, 0.5], [0.2, 0.25, 0.6]]])
        _ = model.plot_scatter_matrix(X=X)
    with pytest.raises(ValueError, match="X should have"):
        X = np.array([[0.1, 0.2, 0.5], [0.2, 0.25, 0.6], [0.3, 0.35, 1.7]])
        _ = model.plot_marginal_distributions(X=X)

    with pytest.raises(ValueError, match="X should be an 2D array"):
        X = np.array([[[0.1, 0.2, 0.5], [0.2, 0.25, 0.6]]])
        _ = model.plot_scatter_matrix(X=X)
    with pytest.raises(ValueError, match="X should have"):
        X = np.array([[0.1, 0.2, 0.5], [0.2, 0.25, 0.6], [0.3, 0.35, 1.7]])
        _ = model.plot_marginal_distributions(X=X)


#
# def _test_fit_re_fit(X):
#     model = VineCopula(
#         max_depth=100,
#         marginal_candidates=[Gaussian()],
#         copula_candidates=[GaussianCopula()],
#         n_jobs=-1,
#         independence_level=1.5,
#     )
#     model.fit(X)
#     sample = model.sample(n_samples=int(5e5), random_state=42)
#     model.fit(sample)
#     sample = model.sample(n_samples=int(5e5), random_state=42)
#     model3 = sk.clone(model)
#     model3.fit(sample)
#     np.testing.assert_array_almost_equal(
#         model.sample(n_samples=5, random_state=42),
#         model3.sample(n_samples=5, random_state=42),
#     )

#
# def _generate_checks_marginals(model):
#     params = ["scale_", "loc_", "dof_", "a_", "b_"]
#     res = []
#     for dist in model.marginal_distributions_:
#         res.append(
#             {
#                 "name": type(dist).__name__,
#                 "params": {
#                     param: round(float(getattr(dist, param)), 5)
#                     for param in params
#                     if hasattr(dist, param)
#                 },
#             }
#         )
#     return res
#
#
# def _generate_checks(model):
#     trees = []
#     for tree in model.trees_:
#         edges = []
#         for edge in tree.edges:
#             if type(edge.copula).__name__ == "StudentTCopula":
#                 params = (
#                     round(float(edge.copula.rho_), 3),
#                     round(float(edge.copula.dof_), 3),
#                 )
#             elif type(edge.copula).__name__ == "GaussianCopula":
#                 params = round(float(edge.copula.rho_), 3)
#             elif type(edge.copula).__name__ == "IndependentCopula":
#                 params = None
#             else:
#                 params = round(float(edge.copula.theta_), 3), edge.copula.rotation_
#             edges.append(
#                 (
#                     edge.cond_sets.conditioned[0],
#                     edge.cond_sets.conditioned[1],
#                     edge.cond_sets.conditioning,
#                     type(edge.copula).__name__,
#                     params,
#                 )
#             )
#         trees.append(edges)
#     return trees
