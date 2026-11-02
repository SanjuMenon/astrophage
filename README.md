# astrophage

Learn thermodynamics through hands-on simulations.

The calculations for P. K. Nag, sections 4.1 through 5.8, are available as an MCP server. Heat added to the system is positive. Work done by the system is positive. `Q = delta_E + W`.

## Run

```powershell
uv sync --extra dev
uv run astrophage
```

Cursor launches it from `.cursor/mcp.json`. Stdio is the transport, so the process should be started by Cursor rather than in a terminal you want to type into.

## Check the book results

```powershell
uv run pytest
```

`book_example` accepts `4.1`, `4.2`, `4.3`, `4.4`, `4.5`, `5.7`, and `5.8`.
