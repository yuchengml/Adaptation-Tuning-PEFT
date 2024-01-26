# Training models
fit_automodel_to_dev:
	python -m scripts.fit fit_automodel_to_dev

fit_automodel_to_post:
	python -m scripts.fit fit_automodel_to_post

fit_peft_w_prefix_tuning_to_dev:
	python -m scripts.fit fit_peft_w_prefix_tuning_to_dev

fit_peft_w_prefix_tuning_to_post:
	python -m scripts.fit fit_peft_w_prefix_tuning_to_post
