.PHONY: build push test clean

build:
    docker build -t bert-moe:latest .

push:
    docker tag bert-moe:latest your-registry/bert-moe:$(VERSION)
    docker push your-registry/bert-moe:$(VERSION)

test:
    pytest tests/ -v

clean:
    docker system prune -f
    rm -rf .pytest_cache

