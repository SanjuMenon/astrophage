"""MCP server for the calculations in P. K. Nag, sections 4.1 through 5.8."""

from __future__ import annotations

from mcp.server import MCPServer

from astrophage import calculations as calc

INSTRUCTIONS = """
Calculations from P. K. Nag, Engineering Thermodynamics, sections 4.1 through 5.8.
Heat added to the system is positive. Work done by the system is positive.
Q = delta_E + W.
Closed-system energies are in kJ. Pressure in kPa times volume in m^3 is kJ.
Flow enthalpies and internal energies are in kJ/kg. Velocity is m/s and
elevation is m, so kinetic energy is V^2/2000 kJ/kg and potential energy is
g Z/1000 kJ/kg, with g = 9.81. Temperatures in the filling results use
T(K) = t(C) + 273, as the section does.
Use book_example with an id of 4.1, 4.2, 4.3, 4.4, 4.5, 5.7, or 5.8 to
reproduce a worked example. Example 4.4 also returns the book's rounded figures.
""".strip()

mcp = MCPServer("astrophage", instructions=INSTRUCTIONS)

TOOLS = (
    calc.cycle_net_work,
    calc.first_law,
    calc.kinetic_energy,
    calc.potential_energy,
    calc.total_stored_energy,
    calc.constant_pressure_work,
    calc.polytropic_state,
    calc.linear_piston_work,
    calc.pv_energy_change,
    calc.constant_volume_heat,
    calc.constant_pressure_heat,
    calc.specific_enthalpy,
    calc.ideal_gas_enthalpy,
    calc.heat_from_enthalpy_change,
    calc.continuity,
    calc.steady_flow_energy,
    calc.multi_stream_balance,
    calc.nozzle,
    calc.throttle,
    calc.adiabatic_turbine,
    calc.adiabatic_compressor,
    calc.heat_exchanger,
    calc.desuperheater,
    calc.euler_residual,
    calc.bernoulli,
    calc.unsteady_control_volume,
    calc.bottle_fill_heat,
    calc.adiabatic_fill_temperature,
    calc.tank_charge,
    calc.ideal_gas_adiabatic_volume_change,
    calc.book_example,
)

for tool in TOOLS:
    mcp.tool()(tool)


def main() -> None:
    mcp.run()
