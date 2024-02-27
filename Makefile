pull:
	~/.envs/emit/bin/python -m pip install --ignore-installed --no-deps "git+https://github.com/csiro-easi/emit"
	
easi-env:
	./scripts/setup-py-env.sh

backup:
	@tar cvzf Data/backup-`date +%Y%m%d`.tar.gz \
		--exclude '**/.ipynb_checkpoints*' \
		--exclude '**/__pycache__*' \
		--exclude Data \
		.
