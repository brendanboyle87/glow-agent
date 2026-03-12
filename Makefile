UV ?= uv
CONFIG ?= configs/glow_smoke.yaml
CONTAINER_CONFIG ?= configs/glow_core.yaml
TRAJECTORY ?= artifacts/trajectories/zork1-episode-000.jsonl

.PHONY: \
	sync test validate-config run-single run-batch inspect-trajectory \
	build-docker run-docker-shell run-zork-docker smoke-test-docker \
	docker-build docker-shell docker-run-single

sync:
	$(UV) sync --extra dev

test:
	$(UV) run pytest

validate-config:
	$(UV) run zork-agent validate-config $(CONFIG)

run-single:
	$(UV) run zork-agent run-single $(CONFIG)

run-batch:
	$(UV) run zork-agent run-batch $(CONFIG)

inspect-trajectory:
	$(UV) run zork-agent inspect-trajectory $(TRAJECTORY)

build-docker:
	docker compose build jericho

run-docker-shell:
	docker compose run --rm jericho bash

run-zork-docker:
	docker compose run --rm jericho uv run zork-agent run-single $(CONTAINER_CONFIG)

smoke-test-docker:
	docker compose run --rm -e SMOKE_CONFIG=$(CONTAINER_CONFIG) jericho smoke-test

docker-build:
	$(MAKE) build-docker

docker-shell:
	$(MAKE) run-docker-shell

docker-run-single:
	$(MAKE) run-zork-docker CONTAINER_CONFIG=$(CONFIG)
