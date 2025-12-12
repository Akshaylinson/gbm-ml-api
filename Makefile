.PHONY: train run build up test lint

train:
	python train.py

run:
	python app.py

build:
	docker build -t gbm-api .

up:
	docker-compose up --build

test:
	pytest -q
