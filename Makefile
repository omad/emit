easi-env:
	./scripts/setup-py-env.sh

backup:
	@tar cvzf Data/backup-`date +%Y%m%d`.tar.gz \
		--exclude '**/.ipynb_checkpoints*' \
		--exclude '**/__pycache__*' \
		--exclude Data \
		.
