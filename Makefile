test:
	pytest

coverage:
	coverage run -m pytest && coverage report --show-missing

lint:
	black liltorch
