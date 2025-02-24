lint:
	ruff format --check
	ruff check

format:
	ruff format
	ruff check --fix
