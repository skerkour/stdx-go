//go:build integration

package s3

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

func TestMinIOIntegration(t *testing.T) {
	endpoint := envOrDefault("S3_INTEGRATION_ENDPOINT", "http://127.0.0.1:9000")
	region := envOrDefault("S3_INTEGRATION_REGION", "us-east-1")
	accessKeyID := envOrDefault("S3_INTEGRATION_ACCESS_KEY_ID", "minioadmin")
	secretAccessKey := envOrDefault("S3_INTEGRATION_SECRET_ACCESS_KEY", "minioadmin")
	sessionToken := os.Getenv("S3_INTEGRATION_SESSION_TOKEN")

	client, err := NewClient(&ClientConfig{
		Endpoint: endpoint,
		Region:   region,
		Credentials: StaticCredentials(
			accessKeyID,
			secretAccessKey,
			sessionToken,
		),
		UsePathStyle: true,
	})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	bucket := fmt.Sprintf("stdx-go-integration-%d", time.Now().UnixNano())
	key := "integration/object.txt"
	body := "hello from stdx-go integration test"

	if _, err := client.CreateBucket(ctx, &CreateBucketInput{Bucket: bucket}); err != nil {
		if strings.Contains(err.Error(), "connect: connection refused") {
			t.Fatalf("create bucket: %v (is MinIO running at %s?)", err, endpoint)
		}
		t.Fatalf("create bucket: %v", err)
	}
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cleanupCancel()

		_, _ = client.DeleteObject(cleanupCtx, &DeleteObjectInput{Bucket: bucket, Key: key})
		_, _ = client.DeleteBucket(cleanupCtx, &DeleteBucketInput{Bucket: bucket})
	})

	if _, err := client.HeadBucket(ctx, &HeadBucketInput{Bucket: bucket}); err != nil {
		t.Fatalf("head bucket: %v", err)
	}

	if _, err := client.PutObject(ctx, &PutObjectInput{
		Bucket:      bucket,
		Key:         key,
		Body:        strings.NewReader(body),
		ContentType: "text/plain",
		Metadata: map[string]string{
			"environment": "integration",
		},
	}); err != nil {
		t.Fatalf("put object: %v", err)
	}

	headObjectOut, err := client.HeadObject(ctx, &HeadObjectInput{Bucket: bucket, Key: key})
	if err != nil {
		t.Fatalf("head object: %v", err)
	}
	if headObjectOut.ContentType != "text/plain" {
		t.Fatalf("unexpected content-type: %q", headObjectOut.ContentType)
	}
	if headObjectOut.Metadata["environment"] != "integration" {
		t.Fatalf("unexpected metadata: %#v", headObjectOut.Metadata)
	}

	getOut, err := client.GetObject(ctx, &GetObjectInput{Bucket: bucket, Key: key})
	if err != nil {
		t.Fatalf("get object: %v", err)
	}
	defer getOut.Body.Close()

	readBody, err := io.ReadAll(getOut.Body)
	if err != nil {
		t.Fatalf("read object body: %v", err)
	}
	if string(readBody) != body {
		t.Fatalf("unexpected object body: %q", string(readBody))
	}
	if getOut.Metadata["environment"] != "integration" {
		t.Fatalf("unexpected get metadata: %#v", getOut.Metadata)
	}

	listOut, err := client.ListObjects(ctx, &ListObjectsInput{Bucket: bucket, Prefix: "integration/"})
	if err != nil {
		t.Fatalf("list objects: %v", err)
	}
	found := false
	for _, object := range listOut.Contents {
		if object.Key == key {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected object %q in list output: %#v", key, listOut.Contents)
	}

	if _, err := client.GetBucketLocation(ctx, &GetBucketLocationInput{Bucket: bucket}); err != nil {
		t.Fatalf("get bucket location: %v", err)
	}

	if _, err := client.DeleteObject(ctx, &DeleteObjectInput{Bucket: bucket, Key: key}); err != nil {
		t.Fatalf("delete object: %v", err)
	}

	if _, err := client.DeleteBucket(ctx, &DeleteBucketInput{Bucket: bucket}); err != nil {
		t.Fatalf("delete bucket: %v", err)
	}
}

func envOrDefault(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}

	return value
}
