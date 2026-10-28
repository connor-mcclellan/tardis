import math

import numpy as np
import numpy.typing as npt
from numba import njit

from tardis.model.geometry.radial1d import (
    NumbaRadial1DGeometry,
)
from tardis.transport.montecarlo import (
    njit_dict_no_parallel,
)
from tardis.transport.montecarlo.configuration.constants import (
    C_SPEED_OF_LIGHT,
    CLOSE_LINE_THRESHOLD,
    MISS_DISTANCE,
    SIGMA_THOMSON,
)
from tardis.transport.montecarlo.packets.radiative_packet import RPacket
from tardis.transport.montecarlo.utils import MonteCarloException


@njit(**njit_dict_no_parallel)
def calculate_distance_boundary(
    r: float, mu: float, r_inner: float, r_outer: float
) -> tuple[float, int]:
    """
    Calculate distance to shell boundary in cm.

    Parameters
    ----------
    r : float
       radial coordinate of the RPacket
    mu : float
       cosine of the direction of movement
    r_inner : float
       inner radius of current shell
    r_outer : float
       outer radius of current shell
    """
    delta_shell = 0
    if mu >= 0.0:
        # direction outward
        distance = math.sqrt(r_outer * r_outer + ((mu * mu - 1.0) * r * r)) - (
            r * mu
        )
        delta_shell = 1
    else:
        # going inward
        check = r_inner * r_inner + (r * r * (mu * mu - 1.0))

        if check >= 0.0:
            # hit inner boundary
            distance = -r * mu - math.sqrt(check)
            delta_shell = -1
        else:
            # miss inner boundary
            distance = math.sqrt(
                r_outer * r_outer + ((mu * mu - 1.0) * r * r)
            ) - (r * mu)
            delta_shell = 1

    return distance, delta_shell


@njit(**njit_dict_no_parallel)
def calculate_distance_line(
    r_packet: RPacket,
    comov_nu: float,
    is_last_line: bool,
    nu_line: float,
    time_explosion: float,
    enable_full_relativity: bool,
) -> float:
    """
    Calculate the homologous-flow distance to the next line resonance.

    Parameters
    ----------
    r_packet : tardis.transport.montecarlo.packets.radiative_packet.RPacket
        Radiative packet being propagated.
    comov_nu : float
        Comoving frequency at the current packet position.
    is_last_line : bool
        Whether the packet has reached the end of the line list.
    nu_line : float
        Rest frequency of the target line.
    time_explosion : float
        Time since explosion in seconds.
    enable_full_relativity : bool
        Whether to use the full-relativity distance calculation.

    Returns
    -------
    float
        Distance to the line resonance in centimeters.
    """
    nu = r_packet.nu

    if is_last_line:
        return MISS_DISTANCE

    nu_diff = comov_nu - nu_line

    # for numerical reasons, if line is too close, we set the distance to 0.
    if abs(nu_diff / nu) < CLOSE_LINE_THRESHOLD:
        return 0.0  # Double check that this is valid. Catches full relativity case.

    if nu_diff >= 0:
        distance = (nu_diff / nu) * C_SPEED_OF_LIGHT * time_explosion
    else:
        raise MonteCarloException("nu difference is less than 0.0")

    if enable_full_relativity:
        return calculate_distance_line_full_relativity(
            nu_line, nu, time_explosion, r_packet
        )
    return distance


@njit(**njit_dict_no_parallel)
def calculate_packet_position(
    initial_radius: float,
    initial_direction_cosine: float,
    distance: float,
) -> tuple[float, float]:
    """Calculate the packet position after a trajectory distance.

    Parameters
    ----------
    initial_radius : float
        Packet radius at the start of the trajectory segment in centimeters.
    initial_direction_cosine : float
        Initial cosine of the packet direction relative to the radial
        direction.
    distance : float
        Distance traveled along the packet trajectory in centimeters.

    Returns
    -------
    tuple[float, float]
        Radius and direction cosine after traveling ``distance``.
    """
    radius = math.sqrt(
        initial_radius * initial_radius
        + distance * distance
        + 2.0 * initial_radius * distance * initial_direction_cosine
    )
    direction_cosine = (
        initial_radius * initial_direction_cosine + distance
    ) / radius
    return radius, direction_cosine


@njit(**njit_dict_no_parallel)
def calculate_projected_velocity_gradient(
    radius: float,
    direction_cosine: float,
    radial_velocity: float,
    radial_velocity_gradient: float,
) -> float:
    """Calculate the velocity gradient projected along a packet trajectory.

    Parameters
    ----------
    radius : float
        Packet radius in centimeters.
    direction_cosine : float
        Cosine of the packet direction relative to the radial direction.
    radial_velocity : float
        Radial flow velocity at ``radius`` in centimeters per second.
    radial_velocity_gradient : float
        Radial derivative of the flow velocity in inverse seconds.

    Returns
    -------
    float
        Projected velocity gradient in inverse seconds.
    """
    return (
        direction_cosine**2 * radial_velocity_gradient
        + (1.0 - direction_cosine**2) * radial_velocity / radius
    )


@njit(fastmath=False, error_model="numpy", parallel=False)
def calculate_projected_velocity_change(
    radius: float,
    direction_cosine: float,
    radial_velocity: float,
    velocity_gradient: float,
    distance: float,
) -> float:
    """Evaluate the projected-velocity change in packet-centered coordinates.

    Parameters
    ----------
    radius : float
        Initial packet radius in centimeters.
    direction_cosine : float
        Initial direction cosine.
    radial_velocity : float
        Radial velocity at the packet position in centimeters per second.
    velocity_gradient : float
        Radial velocity gradient in inverse seconds.
    distance : float
        Distance traveled along the ray in centimeters.

    Returns
    -------
    float
        Change in projected velocity in centimeters per second.
    """
    initial_projected_position = radius * direction_cosine
    final_radius = math.sqrt(
        radius * radius
        + 2.0 * initial_projected_position * distance
        + distance * distance
    )
    radius_change = (
        distance
        * (2.0 * initial_projected_position + distance)
        / (final_radius + radius)
    )
    direction_change = (
        distance - direction_cosine * radius_change
    ) / (radius + radius_change)
    velocity_intercept = radial_velocity - velocity_gradient * radius
    return velocity_gradient * distance + velocity_intercept * direction_change


@njit(fastmath=False, error_model="numpy", parallel=False)
def calculate_distance_line_nonhomologous(
    rpacket: RPacket,
    geometry: NumbaRadial1DGeometry,
    nu_line: float,
    minimum_distance: float = 0.0,
    maximum_distance: float = MISS_DISTANCE,
    monotonic_interval: bool = False,
) -> float:
    """Find the first resonance in a piecewise-linear radial velocity field.

    The projected velocity is monotonic between its gradient zeros. A line
    therefore has a root in one such interval only if its signed residual
    crosses zero there. The lower endpoint is excluded to avoid repeating a
    resonance at the packet or at the preceding interval boundary.

    Parameters
    ----------
    rpacket : RPacket
        Packet defining the ray and lab-frame frequency.
    geometry : NumbaRadial1DGeometry
        Shell radii and radial velocities.
    nu_line : float
        Rest frequency of the line in hertz.
    minimum_distance : float, optional
        Exclusive lower bound in centimeters.
    maximum_distance : float, optional
        Inclusive upper bound in centimeters.
    monotonic_interval : bool, optional
        Whether the supplied bounds already delimit one monotonic interval.

    Returns
    -------
    float
        First forward resonance distance in centimeters, or MISS_DISTANCE.
    """
    shell_id = rpacket.current_shell_id
    radius = rpacket.r
    direction_cosine = rpacket.mu
    velocity_gradient = geometry.velocity_gradient[shell_id]
    radial_velocity = geometry.get_velocity(radius, shell_id)
    doppler_factor = (
        1.0 - radial_velocity * direction_cosine / C_SPEED_OF_LIGHT
    )
    # Subtract frequencies before division. This preserves the zero residual
    # when the represented comoving frequency is exactly the line frequency.
    target_velocity_change = (
        C_SPEED_OF_LIGHT
        * (rpacket.nu * doppler_factor - nu_line)
        / rpacket.nu
    )
    r_inner = geometry.r_inner[shell_id]
    r_outer = geometry.r_outer[shell_id]
    distance_boundary, _ = calculate_distance_boundary(
        radius, direction_cosine, r_inner, r_outer
    )
    interval_end = min(maximum_distance, distance_boundary)
    if interval_end <= minimum_distance:
        return MISS_DISTANCE

    first_turn = MISS_DISTANCE
    second_turn = MISS_DISTANCE
    turn_count = 0
    if not monotonic_interval:
        first_turn, second_turn, turn_count = (
            calculate_projected_gradient_zero_distances(
                rpacket, geometry, interval_end
            )
        )

    shell_width = r_outer - r_inner
    distance_tolerance = CLOSE_LINE_THRESHOLD * shell_width
    interval_start = minimum_distance
    for interval_idx in range(turn_count + 1):
        if interval_idx == 0 and turn_count > 0:
            interval_stop = first_turn
        elif interval_idx == 1 and turn_count > 1:
            interval_stop = second_turn
        else:
            interval_stop = interval_end

        start_residual = calculate_projected_velocity_change(
            radius,
            direction_cosine,
            radial_velocity,
            velocity_gradient,
            interval_start,
        ) - target_velocity_change
        stop_residual = calculate_projected_velocity_change(
            radius,
            direction_cosine,
            radial_velocity,
            velocity_gradient,
            interval_stop,
        ) - target_velocity_change

        # A monotonic interval has no interior root unless its endpoint
        # residuals have opposite signs. In particular, zero at its start is
        # the already visited resonance, not a second interaction.
        if start_residual != 0.0 and (
            stop_residual == 0.0
            or (start_residual > 0.0) != (stop_residual > 0.0)
        ):
            lower_distance = interval_start
            upper_distance = interval_stop
            if stop_residual != 0.0:
                for _ in range(64):
                    middle_distance = 0.5 * (
                        lower_distance + upper_distance
                    )
                    middle_residual = calculate_projected_velocity_change(
                        radius,
                        direction_cosine,
                        radial_velocity,
                        velocity_gradient,
                        middle_distance,
                    ) - target_velocity_change
                    if middle_residual == 0.0:
                        lower_distance = middle_distance
                        upper_distance = middle_distance
                        break
                    if (middle_residual > 0.0) == (start_residual > 0.0):
                        lower_distance = middle_distance
                    else:
                        upper_distance = middle_distance
                    if upper_distance - lower_distance <= distance_tolerance:
                        break
            resonance_distance = 0.5 * (
                lower_distance + upper_distance
            )
            if resonance_distance > interval_start + distance_tolerance:
                return resonance_distance
        interval_start = interval_stop

    return MISS_DISTANCE


@njit(**njit_dict_no_parallel)
def calculate_projected_gradient_zero_distances(
    rpacket: RPacket,
    geometry: NumbaRadial1DGeometry,
    distance_boundary: float,
) -> tuple[float, float, int]:
    """Calculate forward projected-gradient zeros in the current shell.

    Parameters
    ----------
    rpacket : RPacket
        Packet defining the current trajectory.
    geometry : NumbaRadial1DGeometry
        Radial geometry containing the velocity field.
    distance_boundary : float
        Distance to the next shell boundary in centimeters.

    Returns
    -------
    tuple[float, float, int]
        First and second zero distances, followed by the number found. Missing
        distances are returned as ``MISS_DISTANCE``.
    """
    shell_id = rpacket.current_shell_id
    r = rpacket.r
    mu = rpacket.mu
    m = geometry.velocity_gradient[shell_id]
    q = geometry.v_outer[shell_id] - m * geometry.r_outer[shell_id]
    impact_parameter_squared = r * r * (1.0 - mu * mu)

    first_distance = MISS_DISTANCE
    second_distance = MISS_DISTANCE
    zero_count = 0

    if m == 0.0 or q == 0.0 or impact_parameter_squared == 0.0:
        return first_distance, second_distance, zero_count

    # For v(r) = m*r + q, the projected gradient along the ray is
    # m + q*b**2/r**3. Its zeros therefore satisfy r**3 = -q*b**2/m.
    turning_radius_cubed = -q * impact_parameter_squared / m
    if turning_radius_cubed <= 0.0:
        return first_distance, second_distance, zero_count

    turning_radius = turning_radius_cubed ** (1.0 / 3.0)
    shell_width = geometry.r_outer[shell_id] - geometry.r_inner[shell_id]
    distance_tolerance = CLOSE_LINE_THRESHOLD * shell_width
    impact_parameter = math.sqrt(impact_parameter_squared)

    # A zero at the impact parameter only touches zero. A radius-based
    # tolerance could hide two well-separated turns close to the tangent.
    if turning_radius <= impact_parameter:
        return first_distance, second_distance, zero_count

    turning_x_squared = (
        (turning_radius - impact_parameter)
        * (turning_radius + impact_parameter)
    )
    turning_x = math.sqrt(turning_x_squared)
    initial_x = r * mu
    negative_x_distance = -turning_x - initial_x
    positive_x_distance = turning_x - initial_x

    if (
        negative_x_distance > distance_tolerance
        and negative_x_distance < distance_boundary - distance_tolerance
    ):
        first_distance = negative_x_distance
        zero_count = 1

    if (
        positive_x_distance > distance_tolerance
        and positive_x_distance < distance_boundary - distance_tolerance
    ):
        if zero_count == 0:
            first_distance = positive_x_distance
        else:
            second_distance = positive_x_distance
        zero_count += 1

    return first_distance, second_distance, zero_count


@njit(**njit_dict_no_parallel)
def calculate_comoving_frequency_nonhomologous(
    rpacket: RPacket,
    geometry: NumbaRadial1DGeometry,
    distance: float,
) -> float:
    """Calculate packet comoving frequency after a trajectory distance."""
    new_r, new_mu = calculate_packet_position(
        rpacket.r, rpacket.mu, distance
    )
    new_v = geometry.get_velocity(new_r, rpacket.current_shell_id)
    return rpacket.nu * (1.0 - new_v / C_SPEED_OF_LIGHT * new_mu)


@njit(**njit_dict_no_parallel)
def get_line_id_range_nonhomologous(
    line_list_nu: npt.NDArray[np.float64],
    comov_nu_start: float,
    comov_nu_end: float,
) -> tuple[int, int, int]:
    """Return directional line-list bounds for one monotonic path interval.

    The transport line list is ordered by descending frequency. The returned
    range conservatively includes both interval endpoint frequencies; the
    distance bounds reject a resonance at the packet's interval start.
    """
    line_count = len(line_list_nu)
    frequency_tolerance = CLOSE_LINE_THRESHOLD * max(
        abs(comov_nu_start), abs(comov_nu_end)
    )
    minimum_frequency = min(comov_nu_start, comov_nu_end) - frequency_tolerance
    maximum_frequency = max(comov_nu_start, comov_nu_end) + frequency_tolerance

    # Find the first line with frequency <= maximum_frequency.
    lower_idx = 0
    upper_idx = line_count
    while lower_idx < upper_idx:
        middle_idx = (lower_idx + upper_idx) // 2
        if line_list_nu[middle_idx] > maximum_frequency:
            lower_idx = middle_idx + 1
        else:
            upper_idx = middle_idx
    maximum_frequency_line_id = lower_idx

    # Find the first line with frequency < minimum_frequency.
    lower_idx = 0
    upper_idx = line_count
    while lower_idx < upper_idx:
        middle_idx = (lower_idx + upper_idx) // 2
        if line_list_nu[middle_idx] >= minimum_frequency:
            lower_idx = middle_idx + 1
        else:
            upper_idx = middle_idx
    minimum_frequency_stop_line_id = lower_idx

    if comov_nu_end < comov_nu_start:
        return (
            maximum_frequency_line_id,
            minimum_frequency_stop_line_id,
            1,
        )

    return (
        minimum_frequency_stop_line_id - 1,
        maximum_frequency_line_id - 1,
        -1,
    )


@njit(**njit_dict_no_parallel)
def calculate_distance_line_full_relativity(
    nu_line: float,
    nu: float,
    time_explosion: float,
    r_packet: RPacket,
) -> float:
    """Calculate a fully relativistic homologous line-resonance distance."""
    # distance = - mu * r + (ct - nu_r * nu_r * sqrt(ct * ct - (1 + r * r * (1 - mu * mu) * (1 + pow(nu_r, -2))))) / (1 + nu_r * nu_r);
    nu_r = nu_line / nu
    ct = C_SPEED_OF_LIGHT * time_explosion
    distance = -r_packet.mu * r_packet.r + (
        ct
        - nu_r
        * nu_r
        * math.sqrt(
            ct * ct
            - (
                1
                + r_packet.r
                * r_packet.r
                * (1 - r_packet.mu * r_packet.mu)
                * (1 + 1.0 / (nu_r * nu_r))
            )
        )
    ) / (1 + nu_r * nu_r)
    return distance


@njit(**njit_dict_no_parallel)
def calculate_distance_electron(electron_density, tau_event):
    """
    Calculate distance to Thomson Scattering

    Parameters
    ----------
    electron_density : float
    tau_event : float
    """
    # add full_relativity here
    return tau_event / (electron_density * SIGMA_THOMSON)
