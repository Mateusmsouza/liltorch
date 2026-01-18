test:
	coverage run -m pytest tests/
	coverage report -m

coverage:
	coverage run -m pytest && coverage report --show-missing

lint:
	black liltorch

local_docs_server:
	cd documentation && mkdocs serve -w .
