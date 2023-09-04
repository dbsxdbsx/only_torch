default: test lint
all: clean test lint
test:
    cargo test -- --test-threads=1
lint:
    cargo clippy --fix --allow-dirty --allow-staged --allow-no-vcs
clean:
    cargo clean && cargo update

