.PHONY: get_token up all

all: up exec

up:
	docker compose up -d

down:
	docker compose down

exec:
	docker exec -it python bash

clean: down
	yes | docker system prune -a
