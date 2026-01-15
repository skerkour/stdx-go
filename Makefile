# Update dependencies
.PHONY: update_deps
update_deps:
	go get -u ./...
	go mod tidy
	go mod tidy

.PHONY: tidy
tidy:
	go mod tidy

.PHONY: fmt
fmt:
	go fmt ./...

.PHONY: test
test:
	CGO_ENABLED=0 go test ./...
