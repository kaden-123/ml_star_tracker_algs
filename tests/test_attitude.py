import numpy as np
import pytest
from attitude import (
    e_to_q,
    q_to_e,
    e_to_DCM,
    DCM_to_e,
    DCM_to_q,
    q_to_DCM,
)

def test_identity_e_to_q():
    e = np.array([0.0, 0.0, 0.0])
    q = e_to_q(e)
    expected_q = np.array([1.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(q, expected_q, atol=1e-6)


def test_identity_q_to_e():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    e = q_to_e(q)
    expected_e = np.array([0.0, 0.0, 0.0])

    np.testing.assert_allclose(e, expected_e, atol=1e-6)


def test_identity_e_to_DCM():
    e = np.array([0.0, 0.0, 0.0])
    DCM = e_to_DCM(e)
    expected_DCM = np.eye(3)

    np.testing.assert_allclose(DCM, expected_DCM, atol=1e-6)


def test_identity_DCM_to_e():
    DCM = np.eye(3)
    e = DCM_to_e(DCM)
    expected_e = np.array([0.0, 0.0, 0.0])

    np.testing.assert_allclose(e, expected_e, atol=1e-6)


def test_identity_DCM_to_q():
    DCM = np.eye(3)
    q = DCM_to_q(DCM)
    expected_q = np.array([1.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(q, expected_q, atol=1e-6)


def test_identity_q_to_DCM():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    DCM = q_to_DCM(q)
    expected_DCM = np.eye(3)

    np.testing.assert_allclose(DCM, expected_DCM, atol=1e-6)



RIGHT_ANGLE_CASES = [
    pytest.param(
        np.deg2rad([90.0, 90.0, 90.0]),
        np.array([0.7071068, 0, 0.7071068, 0]),
        np.array([[0, 0, -1],
                  [0, 1, 0],
                  [1, 0, 0]]),
        id="90-degrees",
    ),
    pytest.param(
        np.deg2rad([180.0, 180.0, 180.0]),
        np.array([1, 0, 0, 0]),
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]),
                 
        id="180-degrees",
    ),
    pytest.param(
        np.deg2rad([270.0, 270.0, 270.0]),
        np.array([0, 0.7071068, 0, 0.7071068]),
        np.array([[0, 0, 1],
                  [0, -1, 0],
                  [1, 0, 0]]),
        id="270-degrees",
    ),
    pytest.param(
        np.deg2rad([360.0, 360.0, 360.0]),
        np.array([-1, 0, 0, 0]),
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]),
        id="360-degrees",
    ),
]

@pytest.mark.parametrize("e, expected_q, expected_DCM", RIGHT_ANGLE_CASES)
def test_90_deg_mul_e_to_q(e, expected_q, expected_DCM):
    q = e_to_q(e)
    np.testing.assert_allclose(q, expected_q, atol=1e-6)

# no q_to_e cause cannot infinite combos of euler angles


@pytest.mark.parametrize("e, expected_q, expected_DCM", RIGHT_ANGLE_CASES)
def test_right_angle_e_to_DCM(e, expected_q, expected_DCM):
    actual_DCM = e_to_DCM(e)
    np.testing.assert_allclose(actual_DCM, expected_DCM, atol=1e-6)

# no DCM_to_e cause cannot infinite combos of euler angles


@pytest.mark.parametrize("e, expected_q, expected_DCM", RIGHT_ANGLE_CASES)
def test_right_angle_DCM_to_q(e, expected_q, expected_DCM):
    actual_q = DCM_to_q(expected_DCM)

    # q and -q describe the same rotation, so compare DCMs.
    np.testing.assert_allclose(
        q_to_DCM(actual_q),
        expected_DCM,
        atol=1e-6,
    )


@pytest.mark.parametrize("e, expected_q, expected_DCM", RIGHT_ANGLE_CASES)
def test_right_angle_q_to_DCM(e, expected_q, expected_DCM):
    actual_DCM = q_to_DCM(expected_q)
    np.testing.assert_allclose(actual_DCM, expected_DCM, atol=1e-6)



ARBITRARY_CASES = [
    np.deg2rad([20, 30, 40]),
    np.deg2rad([-15, 25, -60]),
    np.deg2rad([120, -35, 75]),
    np.deg2rad([-90, 10, 180]),
]
@pytest.mark.parametrize("e", ARBITRARY_CASES)
def test_arbitrary_values(e):
    q = e_to_q(e)
    dcm = e_to_DCM(e) 
    np.testing.assert_allclose(q_to_DCM(q), dcm, atol=1e-6)
    np.testing.assert_allclose(e_to_DCM(DCM_to_e(dcm)), dcm, atol=1e-6)

