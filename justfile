default: test lint
all: clean test lint
test:
    cargo test -- --test-threads=1
lint:
    cargo fmt
clean:
    cargo clean && cargo update

