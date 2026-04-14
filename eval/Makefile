COMPOSE = docker compose -f eval/docker-compose.yml
IMAGE   = cipher-eval

.PHONY: build ocr haiku sonnet gpt4o gemini all clean

build:
	$(COMPOSE) build ocr

# --- Individual model runs ---
ocr:    build
	$(COMPOSE) run --rm ocr

haiku:  build
	$(COMPOSE) run --rm haiku

sonnet: build
	$(COMPOSE) run --rm sonnet

gpt4o:  build
	$(COMPOSE) run --rm gpt4o

gemini: build
	$(COMPOSE) run --rm gemini

# --- Run all API models (sequential) ---
all: build
	$(COMPOSE) run --rm ocr
	$(COMPOSE) run --rm haiku
	$(COMPOSE) run --rm sonnet
	$(COMPOSE) run --rm gpt4o
	$(COMPOSE) run --rm gemini

# --- Custom model + args ---
# Usage: make run MODEL=haiku ARGS="--levels L1 L2 --max-samples 5"
run: build
	$(COMPOSE) run --rm $(MODEL) $(ARGS)

clean:
	$(COMPOSE) down --rmi local
