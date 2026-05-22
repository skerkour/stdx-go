.PHONY: integration_test
integration_test:
	CGO_ENABLED=0 \
	S3_ENDPOINT=$${S3_ENDPOINT:-http://127.0.0.1:9000} \
	S3_REGION=$${S3_REGION:-auto} \
	S3_ACCESS_KEY_ID=$${S3_ACCESS_KEY_ID:-minioadmin} \
	S3_SECRET_ACCESS_KEY=$${S3_SECRET_ACCESS_KEY:-minioadmin} \
	go test -tags=integration . -run TestMinIOIntegration -count=1
