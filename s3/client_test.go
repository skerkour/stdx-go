package s3

import (
	"bytes"
	"context"
	"encoding/xml"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestMostUsedActions(t *testing.T) {
	expected := []string{
		"GetObject",
		"HeadObject",
		"PutObject",
		"DeleteObject",
		"DeleteObjects",
		"CopyObject",
		"ListObjects",
		"ListBuckets",
		"HeadBucket",
		"CreateBucket",
		"DeleteBucket",
	}

	if !reflect.DeepEqual(MostUsedActions, expected) {
		t.Fatalf("unexpected MostUsedActions: %#v", MostUsedActions)
	}
}

func TestNewClient(t *testing.T) {
	t.Parallel()

	_, err := NewClient(nil)
	if err == nil || err.Error() != "s3: client config is required" {
		t.Fatalf("expected nil config error, got %v", err)
	}

	_, err = NewClient(&ClientConfig{Endpoint: "https://s3.example.com"})
	if err == nil || err.Error() != "s3: credentials are required" {
		t.Fatalf("expected credentials error, got %v", err)
	}

	client, err := NewClient(&ClientConfig{
		Endpoint:    "http://127.0.0.1:9000",
		Credentials: StaticCredentials("access-key", "secret-key", "session-token"),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if client.region != defaultRegion {
		t.Fatalf("expected default region %q, got %q", defaultRegion, client.region)
	}
	if !client.usePathStyle {
		t.Fatal("expected localhost endpoint to use path style")
	}
}

func TestStaticCredentials(t *testing.T) {
	t.Parallel()

	credentials, err := StaticCredentials("access-key", "secret-key", "session-token").Retrieve(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := CredentialsValue{AccessKeyID: "access-key", SecretAccessKey: "secret-key", SessionToken: "session-token"}
	if credentials != expected {
		t.Fatalf("unexpected credentials: %#v", credentials)
	}
}

func TestGetObject(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodGet {
			t.Fatalf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/bucket/folder/object.txt" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.URL.Query().Get("versionId") != "version-1" {
			t.Fatalf("unexpected version id: %s", r.URL.Query().Get("versionId"))
		}
		if r.Header.Get("Range") != "bytes=0-4" {
			t.Fatalf("unexpected range header: %s", r.Header.Get("Range"))
		}

		w.Header().Set("Content-Length", "5")
		w.Header().Set("Content-Type", "text/plain")
		w.Header().Set("ETag", "\"etag-1\"")
		w.Header().Set("Last-Modified", expectedDate.Format(http.TimeFormat))
		w.Header().Set("x-amz-meta-color", "blue")
		w.Header().Set("x-amz-version-id", "version-1")
		_, _ = w.Write([]byte("hello"))
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.GetObject(context.Background(), &GetObjectInput{Bucket: "bucket", Key: "folder/object.txt", Range: "bytes=0-4", VersionID: "version-1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer output.Body.Close()

	body, err := io.ReadAll(output.Body)
	if err != nil {
		t.Fatalf("unexpected read error: %v", err)
	}
	if string(body) != "hello" {
		t.Fatalf("unexpected body: %q", string(body))
	}
	if output.ContentLength != 5 || output.ContentType != "text/plain" || output.ETag != "\"etag-1\"" || output.VersionID != "version-1" {
		t.Fatalf("unexpected object metadata: %#v", output)
	}
	if !output.LastModified.Equal(expectedDate) {
		t.Fatalf("unexpected last modified: %v", output.LastModified)
	}
	if !reflect.DeepEqual(output.Metadata, map[string]string{"color": "blue"}) {
		t.Fatalf("unexpected metadata: %#v", output.Metadata)
	}
}

func TestHeadObject(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodHead {
			t.Fatalf("expected HEAD, got %s", r.Method)
		}

		w.Header().Set("Content-Length", "42")
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("ETag", "\"etag-head\"")
		w.Header().Set("Last-Modified", expectedDate.Format(http.TimeFormat))
		w.Header().Set("x-amz-meta-env", "test")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.HeadObject(context.Background(), &HeadObjectInput{Bucket: "bucket", Key: "file.bin"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if output.ContentLength != 42 || output.ContentType != "application/octet-stream" || output.ETag != "\"etag-head\"" {
		t.Fatalf("unexpected head object output: %#v", output)
	}
	if !reflect.DeepEqual(output.Metadata, map[string]string{"env": "test"}) {
		t.Fatalf("unexpected metadata: %#v", output.Metadata)
	}
}

func TestPutObject(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodPut {
			t.Fatalf("expected PUT, got %s", r.Method)
		}
		if got := r.Header.Get("Content-Type"); got != "text/plain" {
			t.Fatalf("unexpected content-type: %s", got)
		}
		if got := r.Header.Get("x-amz-meta-environment"); got != "test" {
			t.Fatalf("unexpected metadata header: %s", got)
		}
		if got := r.Header.Get("x-amz-content-sha256"); got != "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5" {
			t.Fatalf("unexpected payload hash: %s", got)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("unexpected read error: %v", err)
		}
		if string(body) != "payload" {
			t.Fatalf("unexpected body: %q", string(body))
		}

		w.Header().Set("ETag", "\"etag-put\"")
		w.Header().Set("x-amz-version-id", "version-put")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.PutObject(context.Background(), &PutObjectInput{
		Bucket:      "bucket",
		Key:         "payload.txt",
		Body:        strings.NewReader("payload"),
		ContentType: "text/plain",
		Metadata:    map[string]string{"environment": "test"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if output.ETag != "\"etag-put\"" || output.VersionID != "version-put" {
		t.Fatalf("unexpected put object output: %#v", output)
	}
}

func TestDeleteObject(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodDelete {
			t.Fatalf("expected DELETE, got %s", r.Method)
		}

		w.Header().Set("x-amz-delete-marker", "true")
		w.Header().Set("x-amz-version-id", "version-delete")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.DeleteObject(context.Background(), &DeleteObjectInput{Bucket: "bucket", Key: "file.txt", VersionID: "version-delete"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !output.DeleteMarker || output.VersionID != "version-delete" {
		t.Fatalf("unexpected delete object output: %#v", output)
	}
}

func TestDeleteObjects(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if _, ok := r.URL.Query()["delete"]; !ok {
			t.Fatalf("expected delete query, got %s", r.URL.RawQuery)
		}
		if r.Header.Get("Content-MD5") == "" {
			t.Fatal("expected Content-MD5 header")
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("unexpected read error: %v", err)
		}
		if !strings.Contains(string(body), "<Key>folder/a.txt</Key>") {
			t.Fatalf("unexpected request body: %s", string(body))
		}

		_, _ = io.WriteString(w, `<DeleteResult><Deleted><Key>folder/a.txt</Key></Deleted><Error><Key>folder/b.txt</Key><Code>AccessDenied</Code><Message>denied</Message></Error></DeleteResult>`)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.DeleteObjects(context.Background(), &DeleteObjectsInput{
		Bucket: "bucket",
		Objects: []ObjectIdentifier{
			{Key: "folder/a.txt"},
			{Key: "folder/b.txt", VersionID: "v2"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(output.Deleted) != 1 || output.Deleted[0].Key != "folder/a.txt" {
		t.Fatalf("unexpected deleted objects: %#v", output.Deleted)
	}
	if len(output.Errors) != 1 || output.Errors[0].Code != "AccessDenied" {
		t.Fatalf("unexpected delete errors: %#v", output.Errors)
	}
}

func TestCopyObject(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodPut {
			t.Fatalf("expected PUT, got %s", r.Method)
		}
		if got := r.Header.Get("x-amz-copy-source"); got != "/source-bucket/folder/source.txt" {
			t.Fatalf("unexpected copy source: %s", got)
		}
		if got := r.Header.Get("x-amz-metadata-directive"); got != "COPY" {
			t.Fatalf("unexpected metadata directive: %s", got)
		}

		_, _ = io.WriteString(w, `<CopyObjectResult><LastModified>2026-05-22T07:16:01.000Z</LastModified><ETag>"etag-copy"</ETag></CopyObjectResult>`)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.CopyObject(context.Background(), &CopyObjectInput{
		Bucket:            "bucket",
		Key:               "folder/destination.txt",
		CopySourceBucket:  "source-bucket",
		CopySourceKey:     "folder/source.txt",
		MetadataDirective: "COPY",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if output.ETag != "\"etag-copy\"" {
		t.Fatalf("unexpected copy object output: %#v", output)
	}
}

func TestListObjects(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodGet {
			t.Fatalf("expected GET, got %s", r.Method)
		}
		if got := r.URL.Query().Get("list-type"); got != "2" {
			t.Fatalf("unexpected list-type query: %s", got)
		}
		if got := r.URL.Query().Get("prefix"); got != "folder/" {
			t.Fatalf("unexpected prefix query: %s", got)
		}
		if got := r.URL.Query().Get("continuation-token"); got != "token-1" {
			t.Fatalf("unexpected continuation token: %s", got)
		}

		response := ListObjectsOutput{
			Name:                  "bucket",
			Prefix:                "folder/",
			Delimiter:             "/",
			KeyCount:              1,
			MaxKeys:               1000,
			IsTruncated:           true,
			ContinuationToken:     "token-1",
			NextContinuationToken: "token-2",
			StartAfter:            "folder/a.txt",
			Contents:              []Object{{Key: "folder/a.txt", LastModified: expectedDate, ETag: "\"etag-list\"", Size: 7, StorageClass: "STANDARD"}},
			CommonPrefixes:        []CommonPrefix{{Prefix: "folder/nested/"}},
		}
		if err := xml.NewEncoder(w).Encode(response); err != nil {
			t.Fatalf("unexpected encode error: %v", err)
		}
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.ListObjects(context.Background(), &ListObjectsInput{
		Bucket:            "bucket",
		Prefix:            "folder/",
		Delimiter:         "/",
		ContinuationToken: "token-1",
		MaxKeys:           1000,
		StartAfter:        "folder/a.txt",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if output.Name != "bucket" || output.NextContinuationToken != "token-2" || len(output.Contents) != 1 || len(output.CommonPrefixes) != 1 {
		t.Fatalf("unexpected list objects output: %#v", output)
	}
}

func TestListBuckets(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodGet {
			t.Fatalf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, `<ListAllMyBucketsResult><Owner><ID>owner-id</ID><DisplayName>owner</DisplayName></Owner><Buckets><Bucket><Name>bucket-a</Name><CreationDate>2026-05-22T07:16:01.000Z</CreationDate></Bucket></Buckets></ListAllMyBucketsResult>`)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.ListBuckets(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Owner.ID != "owner-id" || len(output.Buckets) != 1 || output.Buckets[0].Name != "bucket-a" {
		t.Fatalf("unexpected list buckets output: %#v", output)
	}
}

func TestHeadBucket(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodHead {
			t.Fatalf("expected HEAD, got %s", r.Method)
		}
		w.Header().Set("x-amz-bucket-region", "eu-west-1")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.HeadBucket(context.Background(), &HeadBucketInput{Bucket: "bucket"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Region != "eu-west-1" {
		t.Fatalf("unexpected region: %s", output.Region)
	}
}

func TestCreateBucket(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodPut {
			t.Fatalf("expected PUT, got %s", r.Method)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("unexpected read error: %v", err)
		}
		if !strings.Contains(string(body), "<LocationConstraint>eu-west-1</LocationConstraint>") {
			t.Fatalf("unexpected create bucket payload: %s", string(body))
		}
		w.Header().Set("Location", "/bucket")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client, err := NewClient(&ClientConfig{Endpoint: server.URL, Region: "eu-west-1", Credentials: StaticCredentials("access-key", "secret-key", "session-token")})
	if err != nil {
		t.Fatalf("unexpected new client error: %v", err)
	}
	client.now = func() time.Time { return expectedDate }

	output, err := client.CreateBucket(context.Background(), &CreateBucketInput{Bucket: "bucket"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Location != "/bucket" {
		t.Fatalf("unexpected location: %s", output.Location)
	}
}

func TestDeleteBucket(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if r.Method != http.MethodDelete {
			t.Fatalf("expected DELETE, got %s", r.Method)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	if _, err := client.DeleteBucket(context.Background(), &DeleteBucketInput{Bucket: "bucket"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGetBucketLocation(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		if _, ok := r.URL.Query()["location"]; !ok {
			t.Fatalf("expected location query, got %s", r.URL.RawQuery)
		}
		_, _ = io.WriteString(w, `<LocationConstraint>EU</LocationConstraint>`)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	output, err := client.GetBucketLocation(context.Background(), &GetBucketLocationInput{Bucket: "bucket"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Region != "eu-west-1" {
		t.Fatalf("unexpected location region: %s", output.Region)
	}
}

func TestAPIError(t *testing.T) {
	t.Parallel()

	expectedDate := time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertSigned(t, r, "access-key", expectedDate)
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `<Error><Code>NoSuchKey</Code><Message>missing object</Message><Resource>/bucket/missing.txt</Resource><RequestId>request-id</RequestId><HostId>host-id</HostId></Error>`)
	}))
	defer server.Close()

	client := newTestClient(t, server.URL)
	_, err := client.GetObject(context.Background(), &GetObjectInput{Bucket: "bucket", Key: "missing.txt"})
	if err == nil {
		t.Fatal("expected error")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T", err)
	}
	if apiErr.StatusCode != http.StatusNotFound || apiErr.Code != "NoSuchKey" || apiErr.Message != "missing object" {
		t.Fatalf("unexpected API error: %#v", apiErr)
	}
}

func TestCanonicalQueryString(t *testing.T) {
	t.Parallel()

	query := canonicalQueryString(url.Values{"prefix": {"folder/a b"}, "marker": {"one", "two"}})
	if query != "marker=one&marker=two&prefix=folder%2Fa%20b" {
		t.Fatalf("unexpected canonical query: %s", query)
	}
}

func TestPreparePayloadReadSeeker(t *testing.T) {
	t.Parallel()

	body, payloadHash, contentLength, err := preparePayload(bytes.NewReader([]byte("payload")), "", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if payloadHash != "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5" {
		t.Fatalf("unexpected payload hash: %s", payloadHash)
	}
	if contentLength != int64(len("payload")) {
		t.Fatalf("unexpected content length: %d", contentLength)
	}

	data, err := io.ReadAll(body)
	if err != nil {
		t.Fatalf("unexpected read error: %v", err)
	}
	if string(data) != "payload" {
		t.Fatalf("unexpected body: %q", string(data))
	}
}

func newTestClient(t *testing.T, endpoint string) *Client {
	t.Helper()

	client, err := NewClient(&ClientConfig{Endpoint: endpoint, Region: "us-east-1", Credentials: StaticCredentials("access-key", "secret-key", "session-token")})
	if err != nil {
		t.Fatalf("unexpected new client error: %v", err)
	}
	client.now = func() time.Time { return time.Date(2026, time.May, 22, 7, 16, 1, 0, time.UTC) }
	return client
}

func assertSigned(t *testing.T, r *http.Request, accessKeyID string, expectedDate time.Time) {
	t.Helper()

	if got := r.Header.Get("x-amz-date"); got != expectedDate.Format("20060102T150405Z") {
		t.Fatalf("unexpected x-amz-date header: %s", got)
	}
	if got := r.Header.Get("x-amz-security-token"); got != "session-token" {
		t.Fatalf("unexpected security token: %s", got)
	}
	if got := r.Header.Get("Authorization"); !strings.Contains(got, "Credential="+accessKeyID+"/20260522/") || !strings.Contains(got, "/s3/aws4_request") {
		t.Fatalf("unexpected authorization header: %s", got)
	}
	if got := r.Header.Get("x-amz-content-sha256"); got == "" {
		t.Fatal("expected x-amz-content-sha256 header")
	}
}
