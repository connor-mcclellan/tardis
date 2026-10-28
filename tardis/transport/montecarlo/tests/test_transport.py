from collections.abc import Callable

import numpy as np
import pytest
from scipy.optimize import brentq

from tardis.conftest import sync_ndarray_assert_allclose
from tardis.model.geometry.radial1d import NumbaRadial1DGeometry
from tardis.model.geometry.radial1d_homologous import (
    NumbaHomologousRadial1DGeometry,
)
from tardis.opacities.opacity_state_numba import OpacityStateNumba
from tardis.transport.frame_transformations import get_inverse_doppler_factor
from tardis.transport.montecarlo.configuration.constants import C_SPEED_OF_LIGHT
from tardis.transport.montecarlo.estimators.estimators_bulk import (
    init_estimators_bulk,
)
from tardis.transport.montecarlo.estimators.estimators_line import (
    EstimatorsLine,
)
from tardis.transport.montecarlo.modes.homologous_rad_packet_transport import (
    trace_packet as homologous_trace_packet,
)
from tardis.transport.montecarlo.modes.nonhomologous.rad_packet_transport import (
    trace_packet as nonhomologous_trace_packet,
)
from tardis.transport.montecarlo.packets.movement import (
    move_packet_across_shell_boundary,
    move_r_packet,
)
from tardis.transport.montecarlo.packets.radiative_packet import (
    InteractionType,
    PacketStatus,
    RPacket,
)

RTOL = 1.0e-12

NO_LINE_OPACITY = {
    "tau_sobolev": np.zeros((2, 2)),
    "line_list_nu": [3.95e14, 3.90e14],
}
LINE_OPACITY = {
    "tau_sobolev": np.ones((2, 2)) * 100.0,
    "line_list_nu": [3.999e14, 3.998e14],
}
NONHOMOLOGOUS_LINE_OPACITY = {
    "tau_sobolev": np.ones((2, 2)) * 100.0,
    "line_list_nu": [3.80e14, 3.70e14],
}
FALLTHROUGH_OPACITY = {
    "tau_sobolev": np.zeros((2, 2)),
    "line_list_nu": [3.999e14, 3.998e14],
}


@pytest.mark.parametrize("enable_full_relativity", [False, True])
def test_homologous_move_r_packet(
    parametrized_packet,
    homologous_radial_1d_geometry: NumbaHomologousRadial1DGeometry,
    enable_full_relativity: bool,
    regression_data,
) -> None:
    packet = parametrized_packet
    packet.current_shell_id = 1
    estimators = init_estimators_bulk(3)

    move_r_packet(
        packet,
        1.0e13,
        homologous_radial_1d_geometry,
        estimators,
        enable_full_relativity,
    )

    sync_ndarray_assert_allclose(
        regression_data,
        np.array([packet.r, packet.mu]),
        estimators.mean_intensity_total,
        estimators.mean_frequency,
        rtol=RTOL,
    )


def test_nonhomologous_move_r_packet(
    parametrized_packet,
    radial_1d_geometry: NumbaRadial1DGeometry,
    bulk_estimators,
    regression_data,
) -> None:
    packet = parametrized_packet
    packet.current_shell_id = 0

    move_r_packet(
        packet,
        1.0e13,
        radial_1d_geometry,
        bulk_estimators,
        False,
    )

    sync_ndarray_assert_allclose(
        regression_data,
        np.array([packet.r, packet.mu]),
        bulk_estimators.mean_intensity_total,
        bulk_estimators.mean_frequency,
        rtol=RTOL,
    )


def test_homologous_move_r_packet_zero_distance(
    parametrized_packet,
    homologous_radial_1d_geometry: NumbaHomologousRadial1DGeometry,
    regression_data,
) -> None:
    packet = parametrized_packet
    packet.current_shell_id = 1
    estimators = init_estimators_bulk(3)

    move_r_packet(
        packet,
        0.0,
        homologous_radial_1d_geometry,
        estimators,
        False,
    )

    sync_ndarray_assert_allclose(
        regression_data,
        np.array([packet.r, packet.mu]),
        estimators.mean_intensity_total,
        estimators.mean_frequency,
        rtol=RTOL,
    )


def test_nonhomologous_move_r_packet_zero_distance(
    parametrized_packet,
    radial_1d_geometry: NumbaRadial1DGeometry,
    bulk_estimators,
    regression_data,
) -> None:
    packet = parametrized_packet
    packet.current_shell_id = 0

    move_r_packet(
        packet,
        0.0,
        radial_1d_geometry,
        bulk_estimators,
        False,
    )

    sync_ndarray_assert_allclose(
        regression_data,
        np.array([packet.r, packet.mu]),
        bulk_estimators.mean_intensity_total,
        bulk_estimators.mean_frequency,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    (
        "current_shell_id",
        "delta_shell",
        "no_of_shells",
        "expected_status",
        "expected_shell_id",
    ),
    [
        (2, 1, 3, PacketStatus.EMITTED, 2),
        (0, -1, 3, PacketStatus.REABSORBED, 0),
        (1, 0, 3, PacketStatus.IN_PROCESS, 1),
        (1, 1, 3, PacketStatus.IN_PROCESS, 2),
    ],
)
def test_move_packet_across_shell_boundary(
    parametrized_packet,
    current_shell_id: int,
    delta_shell: int,
    no_of_shells: int,
    expected_status: PacketStatus,
    expected_shell_id: int,
) -> None:
    packet = parametrized_packet
    packet.current_shell_id = current_shell_id

    move_packet_across_shell_boundary(packet, delta_shell, no_of_shells)

    assert packet.status == expected_status
    assert packet.current_shell_id == expected_shell_id


@pytest.mark.parametrize(
    (
        "opacity_electron",
        "opacity_state_args",
        "disable_line_scattering",
        "homologous_radial_1d_geometry",
        "expected_interaction_type",
    ),
    [
        (
            1.0e-20,
            NO_LINE_OPACITY,
            False,
            8.0e14,
            InteractionType.BOUNDARY,
        ),
        (
            1.0e-12,
            NO_LINE_OPACITY,
            False,
            8.0e14,
            InteractionType.ESCATTERING,
        ),
        (
            1.0e-20,
            LINE_OPACITY,
            False,
            2.0e16,
            InteractionType.LINE,
        ),
        (
            1.0e-20,
            LINE_OPACITY,
            True,
            2.0e16,
            InteractionType.ESCATTERING,
        ),
    ],
    indirect=["opacity_state_args", "homologous_radial_1d_geometry"],
)
def test_classic_trace_packet(
    parametrized_packet,
    opacity_electron: float,
    classic_opacity_state,
    disable_line_scattering: bool,
    homologous_radial_1d_geometry: NumbaHomologousRadial1DGeometry,
    line_estimators,
    set_seed_fixture,
    expected_interaction_type: InteractionType,
    regression_data,
) -> None:
    set_seed_fixture(1963)

    distance, interaction_type, delta_shell = homologous_trace_packet(
        parametrized_packet,
        homologous_radial_1d_geometry,
        5.2e7,
        classic_opacity_state,
        line_estimators,
        opacity_electron,
        1.0,
        False,
        False,
        disable_line_scattering,
    )

    assert interaction_type == expected_interaction_type
    assert delta_shell == 1
    assert parametrized_packet.next_line_id == (
        1 if disable_line_scattering else 0
    )
    sync_ndarray_assert_allclose(
        regression_data,
        distance,
        line_estimators.mean_intensity_blueward,
        line_estimators.energy_deposition_line_rate,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "parametrized_packet", [{"next_line_id": 2}], indirect=True
)
@pytest.mark.parametrize("homologous_radial_1d_geometry", [2.0e16], indirect=True)
@pytest.mark.parametrize(
    "opacity_state_args",
    [FALLTHROUGH_OPACITY],
    indirect=True,
)
def test_classic_trace_packet_no_line_fallthrough(
    parametrized_packet,
    homologous_radial_1d_geometry: NumbaHomologousRadial1DGeometry,
    classic_opacity_state,
    line_estimators,
    set_seed_fixture,
    regression_data,
) -> None:
    # Start beyond the available line list to lock in the no-line fallthrough.
    set_seed_fixture(1963)

    distance, interaction_type, delta_shell = homologous_trace_packet(
        parametrized_packet,
        homologous_radial_1d_geometry,
        5.2e7,
        classic_opacity_state,
        line_estimators,
        1.0e-12,
        1.0,
        False,
        False,
        False,
    )

    assert interaction_type == InteractionType.ESCATTERING
    assert delta_shell == 1
    assert parametrized_packet.next_line_id == 2
    sync_ndarray_assert_allclose(
        regression_data,
        distance,
        line_estimators.mean_intensity_blueward,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    (
        "chi_continuum",
        "escat_prob",
        "opacity_state_args",
        "disable_line_scattering",
        "homologous_radial_1d_geometry",
        "expected_interaction_type",
    ),
    [
        (
            1.0e-20,
            0.5,
            NO_LINE_OPACITY,
            False,
            8.0e14,
            InteractionType.BOUNDARY,
        ),
        (
            1.0e-12,
            1.0,
            NO_LINE_OPACITY,
            False,
            8.0e14,
            InteractionType.ESCATTERING,
        ),
        (
            1.0e-12,
            0.0,
            NO_LINE_OPACITY,
            False,
            8.0e14,
            InteractionType.CONTINUUM_PROCESS,
        ),
        (
            1.0e-20,
            0.5,
            LINE_OPACITY,
            False,
            2.0e16,
            InteractionType.LINE,
        ),
        (
            1.0e-20,
            0.5,
            LINE_OPACITY,
            True,
            2.0e16,
            InteractionType.ESCATTERING,
        ),
    ],
    indirect=["opacity_state_args", "homologous_radial_1d_geometry"],
)
def test_iip_trace_packet(
    parametrized_packet,
    chi_continuum: float,
    escat_prob: float,
    iip_opacity_state,
    disable_line_scattering: bool,
    homologous_radial_1d_geometry: NumbaHomologousRadial1DGeometry,
    line_estimators,
    set_seed_fixture,
    expected_interaction_type: InteractionType,
    regression_data,
) -> None:
    set_seed_fixture(1963)

    distance, interaction_type, delta_shell = homologous_trace_packet(
        parametrized_packet,
        homologous_radial_1d_geometry,
        5.2e7,
        iip_opacity_state,
        line_estimators,
        chi_continuum,
        escat_prob,
        True,
        False,
        disable_line_scattering,
    )

    assert interaction_type == expected_interaction_type
    assert delta_shell == 1
    assert parametrized_packet.next_line_id == (
        1 if disable_line_scattering else 0
    )
    sync_ndarray_assert_allclose(
        regression_data,
        distance,
        line_estimators.mean_intensity_blueward,
        line_estimators.energy_deposition_line_rate,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "parametrized_packet", [{"next_line_id": 2}], indirect=True
)
@pytest.mark.parametrize("homologous_radial_1d_geometry", [2.0e16], indirect=True)
@pytest.mark.parametrize(
    "opacity_state_args",
    [FALLTHROUGH_OPACITY],
    indirect=True,
)
def test_iip_trace_packet_no_line_fallthrough(
    parametrized_packet,
    homologous_radial_1d_geometry: NumbaHomologousRadial1DGeometry,
    iip_opacity_state,
    line_estimators,
    set_seed_fixture,
    regression_data,
) -> None:
    # Start beyond the available line list to lock in the no-line fallthrough.
    set_seed_fixture(1963)

    distance, interaction_type, delta_shell = homologous_trace_packet(
        parametrized_packet,
        homologous_radial_1d_geometry,
        5.2e7,
        iip_opacity_state,
        line_estimators,
        1.0e-12,
        1.0,
        True,
        False,
        False,
    )

    assert interaction_type == InteractionType.ESCATTERING
    assert delta_shell == 1
    assert parametrized_packet.next_line_id == 2
    sync_ndarray_assert_allclose(
        regression_data,
        distance,
        line_estimators.mean_intensity_blueward,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    (
        "radial_1d_geometry",
        "opacity_electron",
        "opacity_state_args",
        "parametrized_packet",
        "expected_interaction_type",
        "expected_next_line_id",
        "expected_prev_line_id",
    ),
    [
        (
            {"negative_velocity_gradient": False},
            1.0e-20,
            NO_LINE_OPACITY,
            {"next_line_id": 0, "prev_line_id": 0},
            InteractionType.BOUNDARY,
            1,
            0,
        ),
        (
            {},
            1.0e-12,
            NO_LINE_OPACITY,
            {"next_line_id": 0, "prev_line_id": 0},
            InteractionType.ESCATTERING,
            1,
            0,
        ),
        (
            {"r_outer_first_shell": 2.0e16},
            1.0e-20,
            NONHOMOLOGOUS_LINE_OPACITY,
            {"next_line_id": 0, "prev_line_id": 0},
            InteractionType.LINE,
            0,
            -1,
        ),
        (
            {"negative_velocity_gradient": True},
            1.0e-20,
            NO_LINE_OPACITY,
            {"next_line_id": 1, "prev_line_id": 1},
            InteractionType.BOUNDARY,
            1,
            0,
        ),
    ],
    indirect=[
        "radial_1d_geometry",
        "opacity_state_args",
        "parametrized_packet",
    ],
)
def test_nonhomologous_trace_packet(
    radial_1d_geometry: NumbaRadial1DGeometry,
    opacity_electron: float,
    parametrized_packet,
    classic_opacity_state,
    line_estimators,
    set_seed_fixture,
    expected_interaction_type: InteractionType,
    expected_next_line_id: int,
    expected_prev_line_id: int,
    regression_data,
) -> None:
    set_seed_fixture(1963)

    distance, interaction_type, delta_shell = nonhomologous_trace_packet(
        parametrized_packet,
        radial_1d_geometry,
        classic_opacity_state,
        line_estimators,
        opacity_electron,
        False,
        False,
    )

    assert interaction_type == expected_interaction_type
    assert delta_shell == 1
    assert parametrized_packet.next_line_id == expected_next_line_id
    assert parametrized_packet.prev_line_id == expected_prev_line_id
    sync_ndarray_assert_allclose(
        regression_data,
        distance,
        line_estimators.mean_intensity_blueward,
        line_estimators.energy_deposition_line_rate,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "parametrized_packet",
    [{"next_line_id": 2, "prev_line_id": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "radial_1d_geometry",
    [{"r_outer_first_shell": 2.0e16}],
    indirect=True,
)
@pytest.mark.parametrize(
    "opacity_state_args",
    [FALLTHROUGH_OPACITY],
    indirect=True,
)
def test_nonhomologous_trace_packet_no_line_fallthrough(
    parametrized_packet,
    radial_1d_geometry: NumbaRadial1DGeometry,
    classic_opacity_state,
    line_estimators,
    set_seed_fixture,
    regression_data,
) -> None:
    # Start beyond the available line list to lock in the no-line fallthrough.
    set_seed_fixture(1963)

    distance, interaction_type, delta_shell = nonhomologous_trace_packet(
        parametrized_packet,
        radial_1d_geometry,
        classic_opacity_state,
        line_estimators,
        1.0e-12,
        False,
        False,
    )

    assert interaction_type == InteractionType.ESCATTERING
    assert delta_shell == 1
    assert parametrized_packet.next_line_id == 2
    assert parametrized_packet.prev_line_id == 1
    sync_ndarray_assert_allclose(
        regression_data,
        distance,
        line_estimators.mean_intensity_blueward,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    ("mu", "emission_formula"),
    [(0.9, "division"), (0.005, "inverse")],
)
def test_nonhomologous_trace_does_not_reresonate_in_homologous_flow(
    homologous_radial_1d_geometry: NumbaHomologousRadial1DGeometry,
    classic_opacity_state: OpacityStateNumba,
    line_estimators: EstimatorsLine,
    set_seed_fixture: Callable[[int], None],
    mu: float,
    emission_formula: str,
) -> None:
    """An emitted line has no later resonance when v = r / t.

    Claim: comoving frequency falls monotonically along a ray in homology.
    Regime: partial relativity, no continuum opacity, one opaque emitted line.
    Verification: the next event is the outer shell boundary.
    """
    homologous = homologous_radial_1d_geometry
    geometry = NumbaRadial1DGeometry(
        homologous.r_inner,
        homologous.r_outer,
        homologous.v_inner,
        homologous.v_outer,
    )
    opacity_state = classic_opacity_state
    opacity_state.line_list_nu[:] = [4.0e14, 1.0e9]
    opacity_state.tau_sobolev[:] = 0.0
    opacity_state.tau_sobolev[0, :] = 100.0
    radius = 7.5e14
    doppler_factor = 1.0 - radius * mu / (
        C_SPEED_OF_LIGHT * homologous.time_explosion
    )
    line_frequency = opacity_state.line_list_nu[0]
    if emission_formula == "inverse":
        lab_frequency = line_frequency * get_inverse_doppler_factor(
            geometry.get_velocity(radius, 0), mu, False
        )
        assert lab_frequency * doppler_factor > line_frequency
    else:
        lab_frequency = line_frequency / doppler_factor
    packet = RPacket(radius, mu, lab_frequency, 1.0, 42, 0)
    packet.next_line_id = 1
    set_seed_fixture(1963)

    _, interaction_type, _ = nonhomologous_trace_packet(
        packet, geometry, opacity_state, line_estimators, 0.0, False, False
    )

    assert interaction_type == InteractionType.BOUNDARY
    np.testing.assert_array_equal(line_estimators.mean_intensity_blueward, 0.0)
    np.testing.assert_array_equal(line_estimators.energy_deposition_line_rate, 0.0)


def test_nonhomologous_trace_keeps_resolved_blueward_line(
    homologous_radial_1d_geometry: NumbaHomologousRadial1DGeometry,
    classic_opacity_state: OpacityStateNumba,
    line_estimators: EstimatorsLine,
    set_seed_fixture: Callable[[int], None],
) -> None:
    """A line one lab-frequency ulp ahead remains reachable in homology."""
    homologous = homologous_radial_1d_geometry
    geometry = NumbaRadial1DGeometry(
        homologous.r_inner,
        homologous.r_outer,
        homologous.v_inner,
        homologous.v_outer,
    )
    opacity_state = classic_opacity_state
    opacity_state.line_list_nu[:] = [4.0e14, 1.0e9]
    opacity_state.tau_sobolev[:] = 0.0
    opacity_state.tau_sobolev[0, :] = 100.0
    radius, mu = 7.5e14, 0.005
    velocity = geometry.get_velocity(radius, 0)
    doppler_factor = 1.0 - velocity * mu / C_SPEED_OF_LIGHT
    line_frequency = opacity_state.line_list_nu[0]
    center_frequency = line_frequency / doppler_factor
    lab_frequency = center_frequency + np.spacing(center_frequency)
    packet = RPacket(radius, mu, lab_frequency, 1.0, 42, 0)
    packet.next_line_id = 0
    set_seed_fixture(1963)

    distance, interaction_type, _ = nonhomologous_trace_packet(
        packet, geometry, opacity_state, line_estimators, 0.0, False, False
    )

    expected_distance = (
        (lab_frequency * doppler_factor - line_frequency)
        * C_SPEED_OF_LIGHT
        / (lab_frequency * geometry.velocity_gradient[0])
    )
    assert interaction_type == InteractionType.LINE
    np.testing.assert_allclose(distance, expected_distance, rtol=0.0, atol=5.0)


@pytest.mark.parametrize("emission_formula", ["division", "inverse"])
def test_nonhomologous_trace_reresonates_after_gradient_reversal(
    classic_opacity_state: OpacityStateNumba,
    line_estimators: EstimatorsLine,
    set_seed_fixture: Callable[[int], None],
    emission_formula: str,
) -> None:
    """A genuine later crossing of the emitted line remains possible.

    Claim: a nonhomologous flow can reverse the comoving frequency trend.
    Regime: one shell, partial relativity, no continuum opacity.
    Verification: independently bracket the later root of the unsquared
    comoving-frequency equation after the projected-gradient turn.
    """
    geometry = NumbaRadial1DGeometry(
        np.array([1.0e14]),
        np.array([2.0e14]),
        np.array([2.0e9]),
        np.array([5.0e8]),
    )
    opacity_state = classic_opacity_state
    opacity_state.line_list_nu[:] = [4.0e14, 1.0e9]
    opacity_state.tau_sobolev[:] = 0.0
    opacity_state.tau_sobolev[0, 0] = 100.0
    radius, mu = 1.5e14, 0.5
    velocity = geometry.get_velocity(radius, 0)
    line_frequency = opacity_state.line_list_nu[0]
    if emission_formula == "inverse":
        lab_frequency = line_frequency * get_inverse_doppler_factor(
            velocity, mu, False
        )
    else:
        lab_frequency = line_frequency / (
            1.0 - velocity * mu / C_SPEED_OF_LIGHT
        )
    packet = RPacket(
        radius,
        mu,
        lab_frequency,
        1.0,
        42,
        0,
    )
    packet.next_line_id = 1
    set_seed_fixture(1963)

    distance, interaction_type, _ = nonhomologous_trace_packet(
        packet, geometry, opacity_state, line_estimators, 0.0, False, False
    )

    velocity_gradient = geometry.velocity_gradient[0]
    velocity_intercept = geometry.v_outer[0] - velocity_gradient * geometry.r_outer[0]
    impact_parameter_squared = radius**2 * (1.0 - mu**2)
    initial_projected_position = radius * mu
    turning_radius = np.cbrt(
        -velocity_intercept * impact_parameter_squared / velocity_gradient
    )
    turning_distance = (
        np.sqrt(turning_radius**2 - impact_parameter_squared)
        - initial_projected_position
    )
    boundary_distance = (
        np.sqrt(geometry.r_outer[0] ** 2 - impact_parameter_squared)
        - initial_projected_position
    )

    def calculate_frequency_residual(path_distance: float) -> float:
        projected_position = initial_projected_position + path_distance
        path_radius = np.sqrt(projected_position**2 + impact_parameter_squared)
        path_velocity = geometry.v_inner[0] + velocity_gradient * (
            path_radius - geometry.r_inner[0]
        )
        return (
            lab_frequency
            * (1.0 - path_velocity * projected_position / (C_SPEED_OF_LIGHT * path_radius))
            - line_frequency
        )

    assert calculate_frequency_residual(turning_distance) < 0.0
    assert calculate_frequency_residual(boundary_distance) > 0.0
    expected_distance = brentq(
        calculate_frequency_residual, turning_distance, boundary_distance
    )
    assert interaction_type == InteractionType.LINE
    assert packet.next_line_id == 0
    assert turning_distance < distance < boundary_distance
    # At this root, ten frequency ulps divided by |d nu_CMF / ds| are
    # under 100 cm; this also covers the quartic root's rounding error.
    np.testing.assert_allclose(distance, expected_distance, rtol=0.0, atol=100.0)
    assert abs(calculate_frequency_residual(distance)) <= 10.0 * np.spacing(
        line_frequency
    )
