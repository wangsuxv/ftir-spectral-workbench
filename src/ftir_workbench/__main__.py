"""Allow ``python -m ftir_workbench`` to run the unified CLI."""

from .cli import main

raise SystemExit(main())
