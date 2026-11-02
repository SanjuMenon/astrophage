"""Textbook calculations for P. K. Nag, sections 4.1 through 5.8.

Sign convention, used everywhere: heat added to the system is positive, and
work done by the system is positive.

    Q = delta_E + W

Closed-system energies are in kJ. A pressure in kPa times a volume in m^3 is
already in kJ, and the same pairing holds per kilogram.

In the flow tools, specific enthalpy and internal energy are in kJ/kg.
Velocity is in m/s and elevation is in m, so

    kinetic energy   = V^2 / 2000          kJ/kg
    potential energy = g * Z / 1000        kJ/kg

with g = 9.81 m/s^2. The section treats 0 C as 273 K, not 273.15 K.
"""

from __future__ import annotations

import math
from typing import Any


G = 9.81


def _require_one_missing(names: list[str], values: list[float | None]) -> str | None:
    missing = [name for name, value in zip(names, values) if value is None]
    if len(missing) > 1:
        raise ValueError(
            "Leave only one value unset. Missing: " + ", ".join(missing) + "."
        )
    if not missing:
        return None
    return missing[0]


def _ke_kj_per_kg(velocity_m_s: float) -> float:
    return velocity_m_s**2 / 2000.0


def _pe_kj_per_kg(elevation_m: float, g: float = G) -> float:
    return g * elevation_m / 1000.0


def mechanical_energy_kj_per_kg(
    velocity_m_s: float = 0.0,
    elevation_m: float = 0.0,
    g: float = G,
) -> float:
    """Kinetic plus potential energy carried by a stream, in kJ/kg."""
    return _ke_kj_per_kg(velocity_m_s) + _pe_kj_per_kg(elevation_m, g)


def stream_energy_kj_per_kg(
    h_kJ_per_kg: float,
    velocity_m_s: float = 0.0,
    elevation_m: float = 0.0,
    g: float = G,
) -> float:
    """Specific energy carried across a boundary: h + V^2/2 + gZ, in kJ/kg."""
    return h_kJ_per_kg + mechanical_energy_kj_per_kg(velocity_m_s, elevation_m, g)


# ---------------------------------------------------------------------------
# 4.1 and 4.2  First law
# ---------------------------------------------------------------------------


def cycle_net_work(
    heat_transfers_kJ: list[float],
    work_transfers_kJ: list[float] | None = None,
) -> dict[str, float]:
    """Section 4.1. Over a cycle, net work equals net heat.

    Pass every heat transfer of the cycle, positive when heat enters the
    system. Net work done by the system is the sum. If the work transfers are
    also supplied, the residual is net heat minus net work and is zero when
    the cycle obeys the first law.
    """
    net_heat = float(sum(heat_transfers_kJ))
    result: dict[str, float] = {
        "net_heat_kJ": net_heat,
        "net_work_kJ": net_heat,
    }
    if work_transfers_kJ is not None:
        net_work = float(sum(work_transfers_kJ))
        result["supplied_net_work_kJ"] = net_work
        result["residual_kJ"] = net_heat - net_work
    return result


def first_law(
    q_kJ: float | None = None,
    w_kJ: float | None = None,
    delta_e_kJ: float | None = None,
) -> dict[str, float | str]:
    """Section 4.2. Solve Q = delta_E + W for the one missing value.

    Heat added to the system is positive. Work done by the system is positive.
    delta_E is the gain of stored energy. Provide two of the three arguments
    and the third is returned. Provide all three to audit a solved process;
    residual_kJ is Q - delta_E - W.
    """
    missing = _require_one_missing(
        ["q_kJ", "w_kJ", "delta_e_kJ"],
        [q_kJ, w_kJ, delta_e_kJ],
    )
    if missing is None:
        assert q_kJ is not None and w_kJ is not None and delta_e_kJ is not None
        solved = "check"
    elif missing == "q_kJ":
        assert w_kJ is not None and delta_e_kJ is not None
        q_kJ = delta_e_kJ + w_kJ
        solved = "q_kJ"
    elif missing == "w_kJ":
        assert q_kJ is not None and delta_e_kJ is not None
        w_kJ = q_kJ - delta_e_kJ
        solved = "w_kJ"
    else:
        assert q_kJ is not None and w_kJ is not None
        delta_e_kJ = q_kJ - w_kJ
        solved = "delta_e_kJ"
    return {
        "q_kJ": q_kJ,
        "w_kJ": w_kJ,
        "delta_e_kJ": delta_e_kJ,
        "residual_kJ": q_kJ - delta_e_kJ - w_kJ,
        "solved": solved,
    }


# ---------------------------------------------------------------------------
# 4.4  Stored energy
# ---------------------------------------------------------------------------


def kinetic_energy(mass_kg: float, velocity_m_s: float) -> dict[str, float]:
    """Section 4.4. Macroscopic kinetic energy EK = m V^2 / 2, returned in kJ."""
    return {"kinetic_energy_kJ": mass_kg * _ke_kj_per_kg(velocity_m_s)}


def potential_energy(
    mass_kg: float,
    elevation_m: float,
    g: float = G,
) -> dict[str, float]:
    """Section 4.4. Macroscopic potential energy EP = m g Z, returned in kJ."""
    return {"potential_energy_kJ": mass_kg * _pe_kj_per_kg(elevation_m, g)}


def total_stored_energy(
    kinetic_kJ: float = 0.0,
    potential_kJ: float = 0.0,
    internal_kJ: float = 0.0,
) -> dict[str, float]:
    """Section 4.4. E = EK + EP + U for a system with no other stored forms."""
    total = kinetic_kJ + potential_kJ + internal_kJ
    return {
        "kinetic_kJ": kinetic_kJ,
        "potential_kJ": potential_kJ,
        "internal_kJ": internal_kJ,
        "total_kJ": total,
    }


# ---------------------------------------------------------------------------
# Boundary work and the two custom energy formulas used in the examples
# ---------------------------------------------------------------------------


def constant_pressure_work(
    p_kPa: float,
    v1_m3: float,
    v2_m3: float,
) -> dict[str, float]:
    """Quasi-static constant-pressure boundary work, W = p (V2 - V1), in kJ.

    Pressure is absolute and in kPa. Volumes are in m^3. Work is positive when
    the system expands. Example 4.1 uses this with p = 105 kPa.
    """
    return {"work_kJ": p_kPa * (v2_m3 - v1_m3)}


def polytropic_state(
    p1_kPa: float,
    v1_m3: float,
    n: float,
    p2_kPa: float | None = None,
    v2_m3: float | None = None,
) -> dict[str, float]:
    """Quasi-static work for p V^n = constant, plus the unknown end state.

    Provide p1, V1, n, and either p2 or V2. Work is the integral of p dV along
    that path, in kJ. For n = 1 the integral is p1 V1 ln(V2 / V1). Example 4.4
    uses n = 1.2.
    """
    if p2_kPa is None and v2_m3 is None:
        raise ValueError("Provide p2_kPa or v2_m3.")
    if n == 0 and p2_kPa is not None and v2_m3 is None:
        raise ValueError("n = 0 is constant pressure. Provide v2_m3.")
    if p2_kPa is None:
        assert v2_m3 is not None
        if v2_m3 <= 0 or v1_m3 <= 0:
            raise ValueError("Volumes must be positive.")
        p2_kPa = p1_kPa * (v1_m3 / v2_m3) ** n
    elif v2_m3 is None:
        if p2_kPa <= 0 or p1_kPa <= 0:
            raise ValueError("Pressures must be positive on a polytrope.")
        v2_m3 = v1_m3 * (p1_kPa / p2_kPa) ** (1.0 / n)
    else:
        expected = v1_m3 * (p1_kPa / p2_kPa) ** (1.0 / n)
        if not math.isclose(v2_m3, expected, rel_tol=1e-4, abs_tol=1e-6):
            raise ValueError(
                f"End state is not on p V^{n} = constant. V2 on that path is {expected}."
            )
    if abs(n - 1.0) < 1e-12:
        work = p1_kPa * v1_m3 * math.log(v2_m3 / v1_m3)
    else:
        work = (p2_kPa * v2_m3 - p1_kPa * v1_m3) / (1.0 - n)
    return {"p2_kPa": p2_kPa, "v2_m3": v2_m3, "work_kJ": work}


def linear_piston_work(
    p1_kPa: float,
    v1_m3: float,
    p2_kPa: float,
    v2_m3: float,
) -> dict[str, float]:
    """Section example 4.5. Work when p = a + b V, fitted through the end states.

    The integral of (a + b V) dV from V1 to V2 is returned in kJ, together with
    a in kPa and b in kPa/m^3.
    """
    if math.isclose(v2_m3, v1_m3):
        raise ValueError("Volume does not change, so a and b are not fixed by the two states.")
    slope = (p2_kPa - p1_kPa) / (v2_m3 - v1_m3)
    intercept = p1_kPa - slope * v1_m3
    work = intercept * (v2_m3 - v1_m3) + 0.5 * slope * (v2_m3**2 - v1_m3**2)
    return {"a_kPa": intercept, "b_kPa_per_m3": slope, "work_kJ": work}


def pv_energy_change(
    coefficient: float,
    p1_kPa: float,
    v1_m3: float,
    p2_kPa: float,
    v2_m3: float,
) -> dict[str, float]:
    """Change of an internal energy written as U = constant + coefficient * p * V.

    p is in kPa and V is in m^3, so p V is in kJ. The constant, and any term
    proportional to mass, drops out of the change. Example 4.4 uses 3.56.
    Example 4.5 uses 3.15.
    """
    delta = coefficient * (p2_kPa * v2_m3 - p1_kPa * v1_m3)
    return {"delta_u_kJ": delta}


# ---------------------------------------------------------------------------
# 4.5, 4.6, 4.7  Specific heats and enthalpy
# ---------------------------------------------------------------------------


def constant_volume_heat(
    mass_kg: float,
    cv_kJ_per_kgK: float,
    t1: float,
    t2: float,
) -> dict[str, float]:
    """Section 4.5. Heat and internal-energy change at constant volume.

    cv is constant. Temperatures may be C or K; only the difference is used.
    With no work other than p dV, W = 0, so Q = delta_U = m cv (T2 - T1).
    """
    delta_u = mass_kg * cv_kJ_per_kgK * (t2 - t1)
    return {"delta_u_kJ": delta_u, "q_kJ": delta_u, "work_kJ": 0.0}


def constant_pressure_heat(
    mass_kg: float,
    cp_kJ_per_kgK: float,
    t1: float,
    t2: float,
) -> dict[str, float]:
    """Section 4.7. Heat and enthalpy change at constant pressure.

    cp is constant. Q = delta_H = m cp (T2 - T1). This is the closed-system
    result when the only work is p dV.
    """
    delta_h = mass_kg * cp_kJ_per_kgK * (t2 - t1)
    return {"delta_h_kJ": delta_h, "q_kJ": delta_h}


def specific_enthalpy(
    u_kJ_per_kg: float,
    p_kPa: float,
    v_m3_per_kg: float,
) -> dict[str, float]:
    """Section 4.6. h = u + p v, in kJ/kg.

    p in kPa and v in m^3/kg make p v come out in kJ/kg.
    """
    return {"h_kJ_per_kg": u_kJ_per_kg + p_kPa * v_m3_per_kg}


def ideal_gas_enthalpy(
    u_kJ_per_kg: float,
    gas_constant_kJ_per_kgK: float,
    temperature_K: float,
) -> dict[str, float]:
    """Section 4.6. For an ideal gas, h = u + R T, and h depends only on T."""
    return {"h_kJ_per_kg": u_kJ_per_kg + gas_constant_kJ_per_kgK * temperature_K}


def heat_from_enthalpy_change(
    dh_kJ_per_kg: float,
    v_m3_per_kg: float,
    dp_kPa: float,
) -> dict[str, float]:
    """Section 4.7. delta_q = dh - v dp for a closed stationary system.

    The result is specific heat transfer in kJ/kg. v dp uses kPa and m^3/kg.
    At constant pressure, dp is zero and delta_q = dh.
    """
    return {"q_kJ_per_kg": dh_kJ_per_kg - v_m3_per_kg * dp_kPa}


# ---------------------------------------------------------------------------
# 5.3  Continuity and the steady-flow energy equation
# ---------------------------------------------------------------------------


def continuity(
    area_m2: float,
    velocity_m_s: float,
    specific_volume_m3_per_kg: float,
) -> dict[str, float]:
    """Section 5.3.1. Mass flow w = A V / v, in kg/s."""
    if specific_volume_m3_per_kg == 0:
        raise ValueError("Specific volume cannot be zero.")
    return {"mass_flow_kg_s": area_m2 * velocity_m_s / specific_volume_m3_per_kg}


def steady_flow_energy(
    h1_kJ_per_kg: float,
    h2_kJ_per_kg: float | None = None,
    q_kJ_per_kg: float | None = None,
    w_shaft_kJ_per_kg: float | None = None,
    v1_m_s: float = 0.0,
    v2_m_s: float = 0.0,
    z1_m: float = 0.0,
    z2_m: float = 0.0,
    g: float = G,
) -> dict[str, float | str]:
    """Section 5.3.2. Steady flow of one stream, per kilogram.

    q - w_shaft = (h2 + KE2 + PE2) - (h1 + KE1 + PE1). Leave exactly one of
    h2, q, and w_shaft unset. Shaft work is positive when the fluid does work.
    Velocities and elevations default to zero, which is the usual reduction
    for a turbine, a compressor, a throttle, or a heat exchanger.
    """
    missing = _require_one_missing(
        ["h2_kJ_per_kg", "q_kJ_per_kg", "w_shaft_kJ_per_kg"],
        [h2_kJ_per_kg, q_kJ_per_kg, w_shaft_kJ_per_kg],
    )
    e1 = stream_energy_kj_per_kg(h1_kJ_per_kg, v1_m_s, z1_m, g)
    if missing is None:
        assert h2_kJ_per_kg is not None
        assert q_kJ_per_kg is not None and w_shaft_kJ_per_kg is not None
        e2 = stream_energy_kj_per_kg(h2_kJ_per_kg, v2_m_s, z2_m, g)
        solved = "check"
    elif missing == "h2_kJ_per_kg":
        assert q_kJ_per_kg is not None and w_shaft_kJ_per_kg is not None
        e2 = e1 + q_kJ_per_kg - w_shaft_kJ_per_kg
        h2_kJ_per_kg = e2 - mechanical_energy_kj_per_kg(v2_m_s, z2_m, g)
        solved = "h2_kJ_per_kg"
    elif missing == "q_kJ_per_kg":
        assert h2_kJ_per_kg is not None and w_shaft_kJ_per_kg is not None
        e2 = stream_energy_kj_per_kg(h2_kJ_per_kg, v2_m_s, z2_m, g)
        q_kJ_per_kg = w_shaft_kJ_per_kg + (e2 - e1)
        solved = "q_kJ_per_kg"
    else:
        assert h2_kJ_per_kg is not None and q_kJ_per_kg is not None
        e2 = stream_energy_kj_per_kg(h2_kJ_per_kg, v2_m_s, z2_m, g)
        w_shaft_kJ_per_kg = q_kJ_per_kg - (e2 - e1)
        solved = "w_shaft_kJ_per_kg"
    return {
        "h1_kJ_per_kg": h1_kJ_per_kg,
        "h2_kJ_per_kg": h2_kJ_per_kg,
        "q_kJ_per_kg": q_kJ_per_kg,
        "w_shaft_kJ_per_kg": w_shaft_kJ_per_kg,
        "e1_kJ_per_kg": e1,
        "e2_kJ_per_kg": e2,
        "residual_kJ_per_kg": q_kJ_per_kg - w_shaft_kJ_per_kg - (e2 - e1),
        "solved": solved,
    }


def multi_stream_balance(
    inlet_flows_kg_s: list[float],
    inlet_enthalpies_kJ_per_kg: list[float],
    outlet_flows_kg_s: list[float],
    outlet_enthalpies_kJ_per_kg: list[float],
    heat_kW: float = 0.0,
    shaft_work_kW: float = 0.0,
    inlet_velocities_m_s: list[float] | None = None,
    outlet_velocities_m_s: list[float] | None = None,
    inlet_elevations_m: list[float] | None = None,
    outlet_elevations_m: list[float] | None = None,
    g: float = G,
) -> dict[str, float]:
    """Section 5.3. Mass and energy balance for several steady streams.

    Each inlet list must be the same length, and each outlet list must be the
    same length. Heat input and shaft work done by the control volume are in
    kW. A zero residual means the streams, the heat, and the shaft work close
    the steady-flow balance. This is the condenser and desuperheater balance
    before it is reduced.
    """
    inlet_velocities_m_s = inlet_velocities_m_s or [0.0] * len(inlet_flows_kg_s)
    outlet_velocities_m_s = outlet_velocities_m_s or [0.0] * len(outlet_flows_kg_s)
    inlet_elevations_m = inlet_elevations_m or [0.0] * len(inlet_flows_kg_s)
    outlet_elevations_m = outlet_elevations_m or [0.0] * len(outlet_flows_kg_s)
    if not (
        len(inlet_flows_kg_s)
        == len(inlet_enthalpies_kJ_per_kg)
        == len(inlet_velocities_m_s)
        == len(inlet_elevations_m)
    ):
        raise ValueError("Inlet flows, enthalpies, velocities, and elevations differ in length.")
    if not (
        len(outlet_flows_kg_s)
        == len(outlet_enthalpies_kJ_per_kg)
        == len(outlet_velocities_m_s)
        == len(outlet_elevations_m)
    ):
        raise ValueError("Outlet flows, enthalpies, velocities, and elevations differ in length.")

    def _rate(
        flows: list[float],
        enthalpies: list[float],
        velocities: list[float],
        elevations: list[float],
    ) -> float:
        total = 0.0
        for flow, enthalpy, velocity, elevation in zip(flows, enthalpies, velocities, elevations):
            total += flow * stream_energy_kj_per_kg(enthalpy, velocity, elevation, g)
        return total

    mass_in = float(sum(inlet_flows_kg_s))
    mass_out = float(sum(outlet_flows_kg_s))
    energy_in = _rate(
        inlet_flows_kg_s,
        inlet_enthalpies_kJ_per_kg,
        inlet_velocities_m_s,
        inlet_elevations_m,
    )
    energy_out = _rate(
        outlet_flows_kg_s,
        outlet_enthalpies_kJ_per_kg,
        outlet_velocities_m_s,
        outlet_elevations_m,
    )
    return {
        "mass_in_kg_s": mass_in,
        "mass_out_kg_s": mass_out,
        "mass_residual_kg_s": mass_in - mass_out,
        "energy_residual_kW": energy_in + heat_kW - energy_out - shaft_work_kW,
    }


# ---------------------------------------------------------------------------
# 5.4  Devices
# ---------------------------------------------------------------------------


def nozzle(
    h1_kJ_per_kg: float,
    h2_kJ_per_kg: float | None = None,
    v1_m_s: float = 0.0,
    v2_m_s: float | None = None,
    z1_m: float = 0.0,
    z2_m: float = 0.0,
    g: float = G,
) -> dict[str, float | str]:
    """Section 5.4.1. Insulated nozzle or diffuser. No shaft work.

    h1 + V1^2/2 + g Z1 = h2 + V2^2/2 + g Z2. Leave either the exit enthalpy
    or the exit velocity unset. The same relation is a diffuser when the
    velocity falls and the enthalpy rises.
    """
    if (h2_kJ_per_kg is None) == (v2_m_s is None):
        raise ValueError("Leave exactly one of h2_kJ_per_kg and v2_m_s unset.")
    e1 = stream_energy_kj_per_kg(h1_kJ_per_kg, v1_m_s, z1_m, g)
    if v2_m_s is None:
        assert h2_kJ_per_kg is not None
        available = e1 - h2_kJ_per_kg - _pe_kj_per_kg(z2_m, g)
        if available < 0:
            raise ValueError("Exit enthalpy is higher than the inlet energy, so the speed is not real.")
        v2_m_s = math.sqrt(available * 2000.0)
        solved = "v2_m_s"
    else:
        e2_mechanical = mechanical_energy_kj_per_kg(v2_m_s, z2_m, g)
        h2_kJ_per_kg = e1 - e2_mechanical
        solved = "h2_kJ_per_kg"
    return {
        "h1_kJ_per_kg": h1_kJ_per_kg,
        "h2_kJ_per_kg": h2_kJ_per_kg,
        "v1_m_s": v1_m_s,
        "v2_m_s": v2_m_s,
        "solved": solved,
    }


def throttle(
    h1_kJ_per_kg: float,
    v1_m_s: float = 0.0,
    v2_m_s: float = 0.0,
    z1_m: float = 0.0,
    z2_m: float = 0.0,
    g: float = G,
) -> dict[str, float]:
    """Section 5.4.2. Throttling. No heat and no shaft work.

    Exit enthalpy follows from h + V^2/2 + gZ being unchanged. With the usual
    neglect of velocity and elevation, h2 = h1.
    """
    e1 = stream_energy_kj_per_kg(h1_kJ_per_kg, v1_m_s, z1_m, g)
    h2 = e1 - mechanical_energy_kj_per_kg(v2_m_s, z2_m, g)
    return {"h2_kJ_per_kg": h2}


def adiabatic_turbine(
    h1_kJ_per_kg: float,
    h2_kJ_per_kg: float,
    v1_m_s: float = 0.0,
    v2_m_s: float = 0.0,
    z1_m: float = 0.0,
    z2_m: float = 0.0,
    g: float = G,
) -> dict[str, float]:
    """Section 5.4.3. Adiabatic turbine. Work done by the fluid, in kJ/kg.

    With velocity and elevation neglected, w = h1 - h2.
    """
    solved = steady_flow_energy(
        h1_kJ_per_kg,
        h2_kJ_per_kg,
        q_kJ_per_kg=0.0,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        z1_m=z1_m,
        z2_m=z2_m,
        g=g,
    )
    return {"w_shaft_kJ_per_kg": float(solved["w_shaft_kJ_per_kg"])}


def adiabatic_compressor(
    h1_kJ_per_kg: float,
    work_input_kJ_per_kg: float,
    v1_m_s: float = 0.0,
    v2_m_s: float = 0.0,
    z1_m: float = 0.0,
    z2_m: float = 0.0,
    g: float = G,
) -> dict[str, float]:
    """Section 5.4.3. Adiabatic compressor or pump.

    work_input_kJ_per_kg is positive when work is done on the fluid. Exit
    enthalpy rises by that amount when velocity and elevation are negligible.
    The shaft work done by the fluid is the negative of the input.
    """
    solved = steady_flow_energy(
        h1_kJ_per_kg,
        q_kJ_per_kg=0.0,
        w_shaft_kJ_per_kg=-work_input_kJ_per_kg,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        z1_m=z1_m,
        z2_m=z2_m,
        g=g,
    )
    return {
        "h2_kJ_per_kg": float(solved["h2_kJ_per_kg"]),
        "w_shaft_kJ_per_kg": -work_input_kJ_per_kg,
    }


def heat_exchanger(
    w_hot_kg_s: float,
    h_hot_in_kJ_per_kg: float,
    h_hot_out_kJ_per_kg: float,
    w_cold_kg_s: float,
    h_cold_in_kJ_per_kg: float,
    h_cold_out_kJ_per_kg: float | None = None,
) -> dict[str, float]:
    """Section 5.4.4. Two-stream exchanger with no external heat or shaft work.

    The hot stream's enthalpy drop supplies the cold stream's enthalpy rise.
    Leave h_cold_out unset to solve it. This is the steam-condenser balance.
    """
    if w_cold_kg_s == 0:
        raise ValueError("Cold-stream flow cannot be zero.")
    heat_from_hot = w_hot_kg_s * (h_hot_in_kJ_per_kg - h_hot_out_kJ_per_kg)
    if h_cold_out_kJ_per_kg is None:
        h_cold_out_kJ_per_kg = h_cold_in_kJ_per_kg + heat_from_hot / w_cold_kg_s
        residual = 0.0
    else:
        heat_to_cold = w_cold_kg_s * (h_cold_out_kJ_per_kg - h_cold_in_kJ_per_kg)
        residual = heat_from_hot - heat_to_cold
    return {
        "h_cold_out_kJ_per_kg": h_cold_out_kJ_per_kg,
        "heat_from_hot_kW": heat_from_hot,
        "residual_kW": residual,
    }


def desuperheater(
    w_water_kg_s: float,
    h_water_kJ_per_kg: float,
    w_steam_kg_s: float,
    h_steam_kJ_per_kg: float,
) -> dict[str, float]:
    """Section 5.4.4. Spray desuperheater. Water is mixed into steam.

    Mass and enthalpy are both conserved. Kinetic and potential energy are
    neglected, as in the section.
    """
    w_out = w_water_kg_s + w_steam_kg_s
    if w_out == 0:
        raise ValueError("Both inlet flows are zero.")
    h_out = (w_water_kg_s * h_water_kJ_per_kg + w_steam_kg_s * h_steam_kJ_per_kg) / w_out
    return {"w_out_kg_s": w_out, "h_out_kJ_per_kg": h_out}


# ---------------------------------------------------------------------------
# 5.5  Euler and Bernoulli
# ---------------------------------------------------------------------------


def euler_residual(
    specific_volume_m3_per_kg: float,
    dp_Pa: float,
    velocity_m_s: float,
    d_velocity_m_s: float,
    d_elevation_m: float,
    g: float = G,
) -> dict[str, float]:
    """Section 5.5. Euler residual v dp + V dV + g dZ, in J/kg.

    For an inviscid fluid in a pipe with no shaft work the residual is zero.
    Pressure change is in pascal, not kPa, so that the residual is in J/kg.
    """
    residual = (
        specific_volume_m3_per_kg * dp_Pa
        + velocity_m_s * d_velocity_m_s
        + g * d_elevation_m
    )
    return {"residual_J_per_kg": residual}


def bernoulli(
    p1_Pa: float,
    density_kg_per_m3: float,
    v1_m_s: float,
    z1_m: float,
    p2_Pa: float | None = None,
    v2_m_s: float | None = None,
    z2_m: float | None = None,
    g: float = G,
) -> dict[str, float | str]:
    """Section 5.5. Bernoulli equation for an inviscid incompressible fluid.

    p/rho + V^2/2 + g Z is the same at both sections. Leave exactly one of
    p2_Pa, v2_m_s, and z2_m unset. Pressures are in pascal and density is in
    kg/m^3. Provide all three to audit the head residual, in J/kg.
    """
    if density_kg_per_m3 == 0:
        raise ValueError("Density cannot be zero.")
    missing = _require_one_missing(
        ["p2_Pa", "v2_m_s", "z2_m"],
        [p2_Pa, v2_m_s, z2_m],
    )
    head1 = p1_Pa / density_kg_per_m3 + v1_m_s**2 / 2.0 + g * z1_m
    if missing is None:
        assert p2_Pa is not None and v2_m_s is not None and z2_m is not None
        solved = "check"
    elif missing == "p2_Pa":
        assert v2_m_s is not None and z2_m is not None
        p2_Pa = density_kg_per_m3 * (head1 - v2_m_s**2 / 2.0 - g * z2_m)
        solved = "p2_Pa"
    elif missing == "v2_m_s":
        assert p2_Pa is not None and z2_m is not None
        available = head1 - p2_Pa / density_kg_per_m3 - g * z2_m
        if available < 0:
            raise ValueError("The downstream head is already above the upstream head.")
        v2_m_s = math.sqrt(2.0 * available)
        solved = "v2_m_s"
    else:
        assert p2_Pa is not None and v2_m_s is not None
        z2_m = (head1 - p2_Pa / density_kg_per_m3 - v2_m_s**2 / 2.0) / g
        solved = "z2_m"
    head2 = p2_Pa / density_kg_per_m3 + v2_m_s**2 / 2.0 + g * z2_m
    return {
        "p2_Pa": p2_Pa,
        "v2_m_s": v2_m_s,
        "z2_m": z2_m,
        "head1_J_per_kg": head1,
        "head2_J_per_kg": head2,
        "residual_J_per_kg": head1 - head2,
        "solved": solved,
    }


# ---------------------------------------------------------------------------
# 5.6, 5.7, 5.8  Unsteady filling and discharge
# ---------------------------------------------------------------------------


def unsteady_control_volume(
    q_kJ: float,
    w_shaft_kJ: float,
    m_in_kg: float = 0.0,
    h_in_kJ_per_kg: float = 0.0,
    v_in_m_s: float = 0.0,
    z_in_m: float = 0.0,
    m_out_kg: float = 0.0,
    h_out_kJ_per_kg: float = 0.0,
    v_out_m_s: float = 0.0,
    z_out_m: float = 0.0,
    g: float = G,
) -> dict[str, float]:
    """Section 5.6. Integrated mass and energy balance for one filling interval.

    Inlet and outlet states are taken as constant over the interval. The
    stored-energy change is Q - W_shaft plus what the inlet stream brings in
    minus what the outlet stream takes out. With both mass flows at zero this
    is the closed-system first law.
    """
    delta_m = m_in_kg - m_out_kg
    inlet = m_in_kg * stream_energy_kj_per_kg(h_in_kJ_per_kg, v_in_m_s, z_in_m, g)
    outlet = m_out_kg * stream_energy_kj_per_kg(h_out_kJ_per_kg, v_out_m_s, z_out_m, g)
    delta_e = q_kJ - w_shaft_kJ + inlet - outlet
    return {"delta_m_kg": delta_m, "delta_e_kJ": delta_e}


def bottle_fill_heat(
    m1_kg: float,
    u1_kJ_per_kg: float,
    m2_kg: float,
    u2_kJ_per_kg: float,
    h_line_kJ_per_kg: float,
    v_line_m_s: float = 0.0,
) -> dict[str, float]:
    """Section 5.7. Heat added while a bottle is filled from a supply line.

    Q = m2 u2 - m1 u1 - (m2 - m1) (h_line + V_line^2 / 2). The line state does
    not change. Potential energy is neglected, as in the section. An evacuated
    adiabatic bottle with negligible line velocity has u2 = h_line and Q = 0.
    """
    entered = m2_kg - m1_kg
    q = (
        m2_kg * u2_kJ_per_kg
        - m1_kg * u1_kJ_per_kg
        - entered * (h_line_kJ_per_kg + _ke_kj_per_kg(v_line_m_s))
    )
    return {"q_kJ": q, "mass_entered_kg": entered}


def adiabatic_fill_temperature(
    t_line_C: float,
    gamma: float,
) -> dict[str, float]:
    """Sections 5.7 and 5.8. Ideal-gas temperature after an adiabatic fill.

    The bottle or tank starts empty, line kinetic energy is neglected, and no
    heat crosses the wall. Then u2 = h_line, so T2 = gamma * T_line. The
    section uses T(K) = t(C) + 273. For air at 27 C with gamma = 1.4 this is
    420 K, which is 147 C.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    t_line_k = t_line_C + 273.0
    t2_k = gamma * t_line_k
    return {
        "t_line_K": t_line_k,
        "t2_K": t2_k,
        "t2_C": t2_k - 273.0,
    }


def tank_charge(
    m1_kg: float,
    u1_kJ_per_kg: float,
    m2_kg: float,
    h_line_kJ_per_kg: float,
    u2_kJ_per_kg: float | None = None,
) -> dict[str, float | str]:
    """Section 5.8. Charging a tank from a line at constant enthalpy.

    (m2 - m1) h_line = m2 u2 - m1 u1. Leave u2 unset to solve it. An empty
    tank gives u2 = h_line: the entering enthalpy becomes internal energy.
    """
    if m2_kg == 0:
        raise ValueError("Final mass cannot be zero.")
    if u2_kJ_per_kg is None:
        u2_kJ_per_kg = (m1_kg * u1_kJ_per_kg + (m2_kg - m1_kg) * h_line_kJ_per_kg) / m2_kg
        solved = "u2_kJ_per_kg"
    else:
        solved = "check"
    residual = (m2_kg - m1_kg) * h_line_kJ_per_kg - (m2_kg * u2_kJ_per_kg - m1_kg * u1_kJ_per_kg)
    return {
        "u2_kJ_per_kg": u2_kJ_per_kg,
        "residual_kJ": residual,
        "solved": solved,
    }


def ideal_gas_adiabatic_volume_change(
    t1_K: float,
    v1_m3_per_kg: float,
    v2_m3_per_kg: float,
    gamma: float,
) -> dict[str, float]:
    """Section 5.8. Temperature of ideal gas left behind in a discharging tank.

    With no heat, no shaft work, and negligible kinetic and potential energy,
    the gas that remains follows du + p dv = 0. For an ideal gas that is the
    reversible adiabatic relation T2 = T1 (v1 / v2)^(gamma - 1). As the tank
    empties, specific volume rises and temperature falls.
    """
    if v1_m3_per_kg <= 0 or v2_m3_per_kg <= 0:
        raise ValueError("Specific volumes must be positive.")
    t2 = t1_K * (v1_m3_per_kg / v2_m3_per_kg) ** (gamma - 1.0)
    return {"t2_K": t2, "t2_C": t2 - 273.0}


# ---------------------------------------------------------------------------
# Worked examples in the sections
# ---------------------------------------------------------------------------


def book_example(example: str) -> dict[str, Any]:
    """Run a worked example from sections 4.1 through 5.8.

    Accepted ids are 4.1, 4.2, 4.3, 4.4, 4.5, 5.7, and 5.8. Example 4.4
    returns the exact end state and, separately, the values printed in the
    book from its rounded volume of 0.845 m^3.
    """
    runners = {
        "4.1": _example_4_1,
        "4.2": _example_4_2,
        "4.3": _example_4_3,
        "4.4": _example_4_4,
        "4.5": _example_4_5,
        "5.7": _example_5_7,
        "5.8": _example_5_8,
    }
    try:
        return runners[example]()
    except KeyError as exc:
        ids = ", ".join(runners)
        raise ValueError(f"Unknown example {example!r}. Use one of: {ids}.") from exc


def _example_4_1() -> dict[str, Any]:
    work = constant_pressure_work(105.0, 0.30, 0.15)
    solved = first_law(q_kJ=-37.6, w_kJ=work["work_kJ"])
    return {
        "work_kJ": work["work_kJ"],
        "q_kJ": solved["q_kJ"],
        "delta_u_kJ": solved["delta_e_kJ"],
    }


def _example_4_2() -> dict[str, Any]:
    along_acb = first_law(q_kJ=84.0, w_kJ=32.0)
    delta_u = float(along_acb["delta_e_kJ"])
    along_adb = first_law(w_kJ=10.5, delta_e_kJ=delta_u)
    returning = first_law(w_kJ=-21.0, delta_e_kJ=-delta_u)
    along_ad = first_law(w_kJ=10.5, delta_e_kJ=42.0)
    q_adb = float(along_adb["q_kJ"])
    q_ad = float(along_ad["q_kJ"])
    return {
        "delta_u_ab_kJ": delta_u,
        "q_adb_kJ": q_adb,
        "q_return_kJ": returning["q_kJ"],
        "q_ad_kJ": q_ad,
        "q_db_kJ": q_adb - q_ad,
    }


def _example_4_3() -> dict[str, Any]:
    ab = first_law(q_kJ=0.0, w_kJ=2170.0)
    bc = first_law(q_kJ=21000.0, w_kJ=0.0)
    cd_energy = -36600.0
    cd = first_law(q_kJ=-2100.0, delta_e_kJ=cd_energy)
    net_heat_rate = -170.0 * 100.0
    q_da = net_heat_rate - (0.0 + 21000.0 - 2100.0)
    delta_da = -(float(ab["delta_e_kJ"]) + float(bc["delta_e_kJ"]) + cd_energy)
    da = first_law(q_kJ=q_da, delta_e_kJ=delta_da)
    processes = {
        "a-b": {"q_kJ_per_min": 0.0, "w_kJ_per_min": 2170.0, "delta_e_kJ_per_min": ab["delta_e_kJ"]},
        "b-c": {"q_kJ_per_min": 21000.0, "w_kJ_per_min": 0.0, "delta_e_kJ_per_min": bc["delta_e_kJ"]},
        "c-d": {"q_kJ_per_min": -2100.0, "w_kJ_per_min": cd["w_kJ"], "delta_e_kJ_per_min": cd_energy},
        "d-a": {"q_kJ_per_min": q_da, "w_kJ_per_min": da["w_kJ"], "delta_e_kJ_per_min": delta_da},
    }
    return {
        "processes": processes,
        "net_heat_kJ_per_min": net_heat_rate,
        "net_work_kJ_per_min": net_heat_rate,
        "net_work_kW": net_heat_rate / 60.0,
    }


def _example_4_4() -> dict[str, Any]:
    exact = polytropic_state(500.0, 0.22, 1.2, p2_kPa=100.0)
    v2 = exact["v2_m3"]
    delta = pv_energy_change(3.56, 500.0, 0.22, 100.0, v2)
    q = first_law(w_kJ=exact["work_kJ"], delta_e_kJ=delta["delta_u_kJ"])
    other = first_law(q_kJ=30.0, delta_e_kJ=delta["delta_u_kJ"])
    printed_volume = 0.845
    printed_delta = pv_energy_change(3.56, 500.0, 0.22, 100.0, printed_volume)
    printed_work = (100.0 * printed_volume - 500.0 * 0.22) / (1.0 - 1.2)
    printed_other = first_law(q_kJ=30.0, delta_e_kJ=-91.0)
    return {
        "v2_m3": v2,
        "delta_u_kJ": delta["delta_u_kJ"],
        "work_quasistatic_kJ": exact["work_kJ"],
        "q_quasistatic_kJ": q["q_kJ"],
        "work_other_kJ": other["w_kJ"],
        "printed_v2_m3": printed_volume,
        "printed_delta_u_kJ": printed_delta["delta_u_kJ"],
        "printed_work_kJ": printed_work,
        "printed_q_kJ": printed_delta["delta_u_kJ"] + printed_work,
        "printed_work_other_kJ": printed_other["w_kJ"],
    }


def _example_4_5() -> dict[str, Any]:
    piston = linear_piston_work(170.0, 0.03, 400.0, 0.06)
    delta = pv_energy_change(3.15, 170.0, 0.03, 400.0, 0.06)
    heat = first_law(w_kJ=piston["work_kJ"], delta_e_kJ=delta["delta_u_kJ"])
    return {
        "a_kPa": piston["a_kPa"],
        "b_kPa_per_m3": piston["b_kPa_per_m3"],
        "delta_u_kJ": delta["delta_u_kJ"],
        "work_kJ": piston["work_kJ"],
        "q_kJ": heat["q_kJ"],
    }


def _example_5_7() -> dict[str, Any]:
    return adiabatic_fill_temperature(27.0, 1.4)


def _example_5_8() -> dict[str, Any]:
    temperatures = adiabatic_fill_temperature(27.0, 1.4)
    charged = tank_charge(0.0, 0.0, 1.0, h_line_kJ_per_kg=1.0)
    return {
        **temperatures,
        "h_line_kJ_per_kg": 1.0,
        "u2_kJ_per_kg": charged["u2_kJ_per_kg"],
    }
