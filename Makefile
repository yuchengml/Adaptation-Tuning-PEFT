# Training models
fit-all-models-to-post: fit-automodel-to-post fit-peft-w-prefix-tuning-to-post fit-peft-w-p-tuning-to-post fit-peft-w-lora-to-post

fit-automodel-to-dev:
	python -m scripts.fit fit_automodel_to_dev

fit-automodel-to-post:
	python -m scripts.fit fit_automodel_to_post

fit-peft-w-prefix-tuning-to-dev:
	python -m scripts.fit fit_peft_w_prefix_tuning_to_dev

fit-peft-w-prefix-tuning-to-post:
	python -m scripts.fit fit_peft_w_prefix_tuning_to_post

fit-peft-w-p-tuning-to-dev:
	python -m scripts.fit fit_peft_w_p_tuning_to_dev

fit-peft-w-p-tuning-to-post:
	python -m scripts.fit fit_peft_w_p_tuning_to_post

fit-peft_w-lora-to-dev:
	python -m scripts.fit fit_peft_w_lora_to_dev

fit-peft-w-lora-to-post:
	python -m scripts.fit fit_peft_w_lora_to_post
