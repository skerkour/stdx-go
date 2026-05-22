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

.PHONY: test_s3_integration
test_s3_integration:
	CGO_ENABLED=0 \
	S3_INTEGRATION_ENDPOINT=$${S3_INTEGRATION_ENDPOINT:-http://127.0.0.1:9000} \
	S3_INTEGRATION_REGION=$${S3_INTEGRATION_REGION:-us-east-1} \
	S3_INTEGRATION_ACCESS_KEY_ID=$${S3_INTEGRATION_ACCESS_KEY_ID:-minioadmin} \
	S3_INTEGRATION_SECRET_ACCESS_KEY=$${S3_INTEGRATION_SECRET_ACCESS_KEY:-minioadmin} \
	go test -tags=integration ./s3 -run TestMinIOIntegration -count=1
