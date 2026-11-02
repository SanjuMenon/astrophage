"""The worked numbers in sections 4.1 through 5.8."""

import math

import pytest

from astrophage.calculations import (
    adiabatic_compressor,
    adiabatic_fill_temperature,
    adiabatic_turbine,
    bernoulli,
    book_example,
    bottle_fill_heat,
    continuity,
    cycle_net_work,
    desuperheater,
    euler_residual,
    first_law,
    heat_exchanger,
    ideal_gas_adiabatic_volume_change,
    kinetic_energy,
    nozzle,
    polytropic_state,
    potential_energy,
    steady_flow_energy,
    tank_charge,
    throttle,
    unsteady_control_volume,
)


def test_example_4_1():
    result = book_example("4.1")
    assert result["work_kJ"] == pytest.approx(-15.75)
    assert result["delta_u_kJ"] == pytest.approx(-21.85)


def test_example_4_2():
    result = book_example("4.2")
    assert result == pytest.approx(
        {
            "delta_u_ab_kJ": 52.0,
            "q_adb_kJ": 62.5,
            "q_return_kJ": -73.0,
            "q_ad_kJ": 52.5,
            "q_db_kJ": 10.0,
        }
    )


def test_example_4_3():
    result = book_example("4.3")
    assert result["processes"]["c-d"]["w_kJ_per_min"] == pytest.approx(34500)
    assert result["processes"]["d-a"]["q_kJ_per_min"] == pytest.approx(-35900)
    assert result["processes"]["d-a"]["delta_e_kJ_per_min"] == pytest.approx(17770)
    assert result["processes"]["d-a"]["w_kJ_per_min"] == pytest.approx(-53670)
    assert result["net_work_kW"] == pytest.approx(-283.333333)


def test_example_4_4_exact_and_printed():
    result = book_example("4.4")
    expected_volume = 0.22 * 5.0 ** (1.0 / 1.2)
    assert result["v2_m3"] == pytest.approx(expected_volume)
    assert result["v2_m3"] == pytest.approx(0.8412, abs=5e-5)
    assert result["printed_work_kJ"] == pytest.approx(127.5)
    assert result["printed_delta_u_kJ"] == pytest.approx(-90.78)
    assert result["printed_work_other_kJ"] == pytest.approx(121.0)
    # The printed heat uses the unrounded energy change at V2 = 0.845 m^3.
    assert result["printed_q_kJ"] == pytest.approx(36.72)


def test_example_4_5():
    result = book_example("4.5")
    assert result["delta_u_kJ"] == pytest.approx(59.535)
    assert result["work_kJ"] == pytest.approx(8.55)
    assert result["q_kJ"] == pytest.approx(68.085)
    assert result["a_kPa"] == pytest.approx(-60)


def test_examples_5_7_and_5_8():
    filled = book_example("5.7")
    charged = book_example("5.8")
    assert filled["t2_K"] == pytest.approx(420)
    assert filled["t2_C"] == pytest.approx(147)
    assert charged["t2_K"] == pytest.approx(420)
    assert charged["u2_kJ_per_kg"] == pytest.approx(charged["h_line_kJ_per_kg"])


def test_first_law_solves_each_unknown():
    assert first_law(w_kJ=4, delta_e_kJ=6)["q_kJ"] == pytest.approx(10)
    assert first_law(q_kJ=10, delta_e_kJ=6)["w_kJ"] == pytest.approx(4)
    assert first_law(q_kJ=10, w_kJ=4)["delta_e_kJ"] == pytest.approx(6)
    assert first_law(q_kJ=10, w_kJ=4, delta_e_kJ=6)["residual_kJ"] == pytest.approx(0)


def test_cycle_matches_section_4_1():
    result = cycle_net_work([10, -4, 1], [7])
    assert result["net_heat_kJ"] == pytest.approx(7)
    assert result["residual_kJ"] == pytest.approx(0)


def test_polytropic_isothermal_work():
    result = polytropic_state(100, 1, 1, v2_m3=2)
    assert result["p2_kPa"] == pytest.approx(50)
    assert result["work_kJ"] == pytest.approx(100 * math.log(2))


def test_stored_energy_and_flow_reductions():
    assert kinetic_energy(2, 10)["kinetic_energy_kJ"] == pytest.approx(0.1)
    assert potential_energy(2, 10)["potential_energy_kJ"] == pytest.approx(0.1962)
    assert nozzle(h1_kJ_per_kg=10, h2_kJ_per_kg=9, v1_m_s=0)["v2_m_s"] == pytest.approx(
        math.sqrt(2000)
    )
    assert throttle(100)["h2_kJ_per_kg"] == pytest.approx(100)
    assert adiabatic_turbine(100, 60)["w_shaft_kJ_per_kg"] == pytest.approx(40)
    assert adiabatic_compressor(100, 25)["h2_kJ_per_kg"] == pytest.approx(125)


def test_steady_flow_includes_kinetic_energy():
    solved = steady_flow_energy(
        h1_kJ_per_kg=100,
        h2_kJ_per_kg=90,
        q_kJ_per_kg=0,
        v1_m_s=0,
        v2_m_s=100,
    )
    # w = q - (delta h + V2^2/2000) = 0 - (-10 + 5) = 5
    assert solved["w_shaft_kJ_per_kg"] == pytest.approx(5)


def test_devices_and_pipe_mechanics():
    cold = heat_exchanger(2, 3000, 2000, 10, 100)
    assert cold["h_cold_out_kJ_per_kg"] == pytest.approx(300)
    mixed = desuperheater(1, 200, 1, 3000)
    assert mixed == pytest.approx({"w_out_kg_s": 2, "h_out_kJ_per_kg": 1600})
    assert continuity(0.1, 10, 0.5)["mass_flow_kg_s"] == pytest.approx(2)
    assert euler_residual(1, -100, 10, 10, 0)["residual_J_per_kg"] == pytest.approx(0)
    downstream = bernoulli(100, 1, 0, 0, v2_m_s=0, z2_m=10)
    assert downstream["p2_Pa"] == pytest.approx(100 - 9.81 * 10)


def test_unsteady_limits():
    closed = unsteady_control_volume(q_kJ=10, w_shaft_kJ=3)
    assert closed["delta_e_kJ"] == pytest.approx(7)
    assert closed["delta_m_kg"] == pytest.approx(0)
    evacuated = bottle_fill_heat(0, 0, 2, 80, 80)
    assert evacuated["q_kJ"] == pytest.approx(0)
    charged = tank_charge(1, 10, 3, 50)
    assert charged["u2_kJ_per_kg"] == pytest.approx(110 / 3)
    cooled = ideal_gas_adiabatic_volume_change(300, 1, 2, 1.4)
    assert cooled["t2_K"] == pytest.approx(300 * 0.5**0.4)
    assert adiabatic_fill_temperature(27, 1.4)["t2_C"] == pytest.approx(147)
