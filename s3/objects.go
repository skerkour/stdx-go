package s3

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/base64"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

type GetObjectInput struct {
	Bucket    string
	Key       string
	Range     string
	VersionID string
}

type GetObjectOutput struct {
	Body          io.ReadCloser
	ContentLength int64
	ContentType   string
	ETag          string
	LastModified  time.Time
	Metadata      map[string]string
	VersionID     string
}

type HeadObjectInput struct {
	Bucket    string
	Key       string
	VersionID string
}

type HeadObjectOutput struct {
	ContentLength int64
	ContentType   string
	ETag          string
	LastModified  time.Time
	Metadata      map[string]string
	VersionID     string
}

type PutObjectInput struct {
	Bucket        string
	Key           string
	Body          io.Reader
	ContentLength int64
	ContentType   string
	ContentMD5    string
	ContentSHA256 string
	Metadata      map[string]string
}

type PutObjectOutput struct {
	ETag      string
	VersionID string
}

type CreateMultipartUploadInput struct {
	Bucket      string
	Key         string
	ContentType string
	Metadata    map[string]string
}

type CreateMultipartUploadOutput struct {
	Bucket   string
	Key      string
	UploadID string
}

type UploadPartInput struct {
	Bucket        string
	Key           string
	UploadID      string
	PartNumber    int
	Body          io.Reader
	ContentLength int64
	ContentMD5    string
	ContentSHA256 string
}

type UploadPartOutput struct {
	ETag string
}

type CompleteMultipartUploadInput struct {
	Bucket   string
	Key      string
	UploadID string
	Parts    []CompletedPart
}

type CompletedPart struct {
	PartNumber int    `xml:"PartNumber"`
	ETag       string `xml:"ETag"`
}

type CompleteMultipartUploadOutput struct {
	Location string
	Bucket   string
	Key      string
	ETag     string
}

type AbortMultipartUploadInput struct {
	Bucket   string
	Key      string
	UploadID string
}

type AbortMultipartUploadOutput struct{}

type DeleteObjectInput struct {
	Bucket    string
	Key       string
	VersionID string
}

type DeleteObjectOutput struct {
	DeleteMarker bool
	VersionID    string
}

type DeleteObjectsInput struct {
	Bucket  string
	Objects []ObjectIdentifier
	Quiet   bool
}

type ObjectIdentifier struct {
	Key       string
	VersionID string
}

type DeleteObjectsOutput struct {
	Deleted []DeletedObject `xml:"Deleted"`
	Errors  []DeleteError   `xml:"Error"`
}

type DeletedObject struct {
	Key          string `xml:"Key"`
	VersionID    string `xml:"VersionId"`
	DeleteMarker bool   `xml:"DeleteMarker"`
}

type DeleteError struct {
	Key       string `xml:"Key"`
	VersionID string `xml:"VersionId"`
	Code      string `xml:"Code"`
	Message   string `xml:"Message"`
}

type CopyObjectInput struct {
	Bucket            string
	Key               string
	CopySourceBucket  string
	CopySourceKey     string
	MetadataDirective string
	StorageClass      string
	ChecksumAlgorithm string
	IfMatchETag       string
	IfNoneMatchETag   string
	IfModifiedSince   time.Time
	IfUnmodifiedSince time.Time
}

type CopyObjectOutput struct {
	ETag         string
	LastModified time.Time
}

type ListObjectsInput struct {
	Bucket            string
	Prefix            string
	Delimiter         string
	ContinuationToken string
	MaxKeys           int
	StartAfter        string
}

type ListObjectsOutput struct {
	Name                  string
	Prefix                string
	Delimiter             string
	KeyCount              int
	MaxKeys               int
	IsTruncated           bool
	ContinuationToken     string
	NextContinuationToken string
	StartAfter            string
	Contents              []Object
	CommonPrefixes        []CommonPrefix
}

type Object struct {
	Key          string    `xml:"Key"`
	LastModified time.Time `xml:"LastModified"`
	ETag         string    `xml:"ETag"`
	Size         int64     `xml:"Size"`
	StorageClass string    `xml:"StorageClass"`
}

type CommonPrefix struct {
	Prefix string `xml:"Prefix"`
}

func (client *Client) GetObject(ctx context.Context, input *GetObjectInput) (*GetObjectOutput, error) {
	if err := validateBucketAndKey(inputBucketKey(input)); err != nil {
		return nil, err
	}

	query := url.Values{}
	if input.VersionID != "" {
		query.Set("versionId", input.VersionID)
	}

	headers := http.Header{}
	if input.Range != "" {
		headers.Set("Range", input.Range)
	}

	res, err := client.do(ctx, http.MethodGet, input.Bucket, input.Key, query, headers, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	output := &GetObjectOutput{Body: res.Body}
	populateObjectHeaders(output, res.Header)
	return output, nil
}

func (client *Client) HeadObject(ctx context.Context, input *HeadObjectInput) (*HeadObjectOutput, error) {
	if err := validateBucketAndKey(inputBucketKey(input)); err != nil {
		return nil, err
	}

	query := url.Values{}
	if input.VersionID != "" {
		query.Set("versionId", input.VersionID)
	}

	res, err := client.do(ctx, http.MethodHead, input.Bucket, input.Key, query, nil, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	output := &HeadObjectOutput{}
	populateObjectHeaders(output, res.Header)
	return output, nil
}

func (client *Client) PutObject(ctx context.Context, input *PutObjectInput) (*PutObjectOutput, error) {
	if err := validateBucketAndKey(inputBucketKey(input)); err != nil {
		return nil, err
	}
	if input.Body == nil {
		return nil, errors.New("s3: put object body is required")
	}

	body, payloadHash, contentLength, err := preparePayload(input.Body, input.ContentSHA256, input.ContentLength)
	if err != nil {
		return nil, err
	}

	headers := http.Header{}
	if input.ContentType != "" {
		headers.Set("Content-Type", input.ContentType)
	}
	if input.ContentMD5 != "" {
		headers.Set("Content-MD5", input.ContentMD5)
	}
	for key, value := range input.Metadata {
		headers.Set(metadataHeaderKey+strings.ToLower(key), value)
	}

	res, err := client.do(ctx, http.MethodPut, input.Bucket, input.Key, nil, headers, body, contentLength, payloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	return &PutObjectOutput{
		ETag:      res.Header.Get("ETag"),
		VersionID: res.Header.Get("x-amz-version-id"),
	}, nil
}

func (client *Client) CreateMultipartUpload(ctx context.Context, input *CreateMultipartUploadInput) (*CreateMultipartUploadOutput, error) {
	if err := validateBucketAndKey(inputBucketKey(input)); err != nil {
		return nil, err
	}

	headers := http.Header{}
	if input.ContentType != "" {
		headers.Set("Content-Type", input.ContentType)
	}
	for key, value := range input.Metadata {
		headers.Set(metadataHeaderKey+strings.ToLower(key), value)
	}

	query := url.Values{"uploads": {""}}
	res, err := client.do(ctx, http.MethodPost, input.Bucket, input.Key, query, headers, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	var output struct {
		Bucket   string `xml:"Bucket"`
		Key      string `xml:"Key"`
		UploadID string `xml:"UploadId"`
	}
	if err := xml.NewDecoder(res.Body).Decode(&output); err != nil {
		return nil, fmt.Errorf("s3: decoding create multipart upload response: %w", err)
	}

	return &CreateMultipartUploadOutput{
		Bucket:   output.Bucket,
		Key:      output.Key,
		UploadID: output.UploadID,
	}, nil
}

func (client *Client) UploadPart(ctx context.Context, input *UploadPartInput) (*UploadPartOutput, error) {
	if err := validateBucketAndKey(inputBucketKey(input)); err != nil {
		return nil, err
	}
	if input.UploadID == "" {
		return nil, errors.New("s3: upload id is required")
	}
	if input.PartNumber <= 0 {
		return nil, errors.New("s3: part number must be greater than zero")
	}
	if input.Body == nil {
		return nil, errors.New("s3: upload part body is required")
	}

	body, payloadHash, contentLength, err := preparePayload(input.Body, input.ContentSHA256, input.ContentLength)
	if err != nil {
		return nil, err
	}

	headers := http.Header{}
	if input.ContentMD5 != "" {
		headers.Set("Content-MD5", input.ContentMD5)
	}
	query := url.Values{
		"partNumber": {strconv.Itoa(input.PartNumber)},
		"uploadId":   {input.UploadID},
	}
	res, err := client.do(ctx, http.MethodPut, input.Bucket, input.Key, query, headers, body, contentLength, payloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	return &UploadPartOutput{ETag: res.Header.Get("ETag")}, nil
}

func (client *Client) CompleteMultipartUpload(ctx context.Context, input *CompleteMultipartUploadInput) (*CompleteMultipartUploadOutput, error) {
	if err := validateBucketAndKey(inputBucketKey(input)); err != nil {
		return nil, err
	}
	if input.UploadID == "" {
		return nil, errors.New("s3: upload id is required")
	}
	if len(input.Parts) == 0 {
		return nil, errors.New("s3: at least one completed part is required")
	}
	for _, part := range input.Parts {
		if part.PartNumber <= 0 {
			return nil, errors.New("s3: completed part number must be greater than zero")
		}
		if strings.TrimSpace(part.ETag) == "" {
			return nil, errors.New("s3: completed part etag is required")
		}
	}

	bodyPayload, err := xml.Marshal(struct {
		XMLName xml.Name        `xml:"CompleteMultipartUpload"`
		Parts   []CompletedPart `xml:"Part"`
	}{
		Parts: input.Parts,
	})
	if err != nil {
		return nil, fmt.Errorf("s3: encoding complete multipart upload request: %w", err)
	}

	query := url.Values{"uploadId": {input.UploadID}}
	headers := http.Header{}
	headers.Set("Content-Type", "application/xml")
	payloadHash := hashSHA256Hex(bodyPayload)
	res, err := client.do(ctx, http.MethodPost, input.Bucket, input.Key, query, headers, bytes.NewReader(bodyPayload), int64(len(bodyPayload)), payloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	var output struct {
		Location string `xml:"Location"`
		Bucket   string `xml:"Bucket"`
		Key      string `xml:"Key"`
		ETag     string `xml:"ETag"`
	}
	if err := xml.NewDecoder(res.Body).Decode(&output); err != nil {
		return nil, fmt.Errorf("s3: decoding complete multipart upload response: %w", err)
	}

	return &CompleteMultipartUploadOutput{
		Location: output.Location,
		Bucket:   output.Bucket,
		Key:      output.Key,
		ETag:     output.ETag,
	}, nil
}

func (client *Client) AbortMultipartUpload(ctx context.Context, input *AbortMultipartUploadInput) (*AbortMultipartUploadOutput, error) {
	if err := validateBucketAndKey(inputBucketKey(input)); err != nil {
		return nil, err
	}
	if input.UploadID == "" {
		return nil, errors.New("s3: upload id is required")
	}

	query := url.Values{"uploadId": {input.UploadID}}
	res, err := client.do(ctx, http.MethodDelete, input.Bucket, input.Key, query, nil, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	return &AbortMultipartUploadOutput{}, nil
}

func (client *Client) DeleteObject(ctx context.Context, input *DeleteObjectInput) (*DeleteObjectOutput, error) {
	if err := validateBucketAndKey(inputBucketKey(input)); err != nil {
		return nil, err
	}

	query := url.Values{}
	if input.VersionID != "" {
		query.Set("versionId", input.VersionID)
	}

	res, err := client.do(ctx, http.MethodDelete, input.Bucket, input.Key, query, nil, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	return &DeleteObjectOutput{
		DeleteMarker: res.Header.Get("x-amz-delete-marker") == "true",
		VersionID:    res.Header.Get("x-amz-version-id"),
	}, nil
}

func (client *Client) DeleteObjects(ctx context.Context, input *DeleteObjectsInput) (*DeleteObjectsOutput, error) {
	if input == nil {
		return nil, errors.New("s3: delete objects input is required")
	}
	if input.Bucket == "" {
		return nil, errors.New("s3: bucket is required")
	}
	if len(input.Objects) == 0 {
		return nil, errors.New("s3: at least one object is required")
	}
	for _, object := range input.Objects {
		if object.Key == "" {
			return nil, errors.New("s3: object key is required")
		}
	}

	bodyPayload, err := xml.Marshal(struct {
		XMLName xml.Name           `xml:"Delete"`
		Quiet   bool               `xml:"Quiet,omitempty"`
		Objects []objectIdentifier `xml:"Object"`
	}{
		Quiet:   input.Quiet,
		Objects: toDeleteIdentifiers(input.Objects),
	})
	if err != nil {
		return nil, fmt.Errorf("s3: encoding delete objects request: %w", err)
	}

	payloadHash := hashSHA256Hex(bodyPayload)
	contentMD5 := md5.Sum(bodyPayload)
	headers := http.Header{}
	headers.Set("Content-Type", "application/xml")
	headers.Set("Content-MD5", base64.StdEncoding.EncodeToString(contentMD5[:]))

	query := url.Values{"delete": {""}}
	res, err := client.do(ctx, http.MethodPost, input.Bucket, "", query, headers, bytes.NewReader(bodyPayload), int64(len(bodyPayload)), payloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	var output DeleteObjectsOutput
	if err := xml.NewDecoder(res.Body).Decode(&output); err != nil {
		return nil, fmt.Errorf("s3: decoding delete objects response: %w", err)
	}

	return &output, nil
}

func (client *Client) CopyObject(ctx context.Context, input *CopyObjectInput) (*CopyObjectOutput, error) {
	if err := validateBucketAndKey(inputBucketKey(input)); err != nil {
		return nil, err
	}
	if input.CopySourceBucket == "" {
		return nil, errors.New("s3: copy source bucket is required")
	}
	if input.CopySourceKey == "" {
		return nil, errors.New("s3: copy source key is required")
	}

	headers := http.Header{}
	headers.Set("x-amz-copy-source", "/"+uriEncode(strings.TrimPrefix(input.CopySourceBucket, "/"), true)+"/"+uriEncode(strings.TrimPrefix(input.CopySourceKey, "/"), false))
	if input.MetadataDirective != "" {
		headers.Set("x-amz-metadata-directive", input.MetadataDirective)
	}
	if input.StorageClass != "" {
		headers.Set("x-amz-storage-class", input.StorageClass)
	}
	if input.ChecksumAlgorithm != "" {
		headers.Set("x-amz-checksum-algorithm", input.ChecksumAlgorithm)
	}
	if input.IfMatchETag != "" {
		headers.Set("x-amz-copy-source-if-match", input.IfMatchETag)
	}
	if input.IfNoneMatchETag != "" {
		headers.Set("x-amz-copy-source-if-none-match", input.IfNoneMatchETag)
	}
	if !input.IfModifiedSince.IsZero() {
		headers.Set("x-amz-copy-source-if-modified-since", input.IfModifiedSince.UTC().Format(http.TimeFormat))
	}
	if !input.IfUnmodifiedSince.IsZero() {
		headers.Set("x-amz-copy-source-if-unmodified-since", input.IfUnmodifiedSince.UTC().Format(http.TimeFormat))
	}

	res, err := client.do(ctx, http.MethodPut, input.Bucket, input.Key, nil, headers, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	var result struct {
		XMLName      xml.Name  `xml:"CopyObjectResult"`
		ETag         string    `xml:"ETag"`
		LastModified time.Time `xml:"LastModified"`
	}
	if err := xml.NewDecoder(res.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("s3: decoding copy object response: %w", err)
	}

	return &CopyObjectOutput{
		ETag:         result.ETag,
		LastModified: result.LastModified,
	}, nil
}

func (client *Client) ListObjects(ctx context.Context, input *ListObjectsInput) (*ListObjectsOutput, error) {
	if input == nil {
		return nil, errors.New("s3: list objects input is required")
	}
	if input.Bucket == "" {
		return nil, errors.New("s3: bucket is required")
	}

	query := url.Values{}
	query.Set("list-type", "2")
	if input.Prefix != "" {
		query.Set("prefix", input.Prefix)
	}
	if input.Delimiter != "" {
		query.Set("delimiter", input.Delimiter)
	}
	if input.ContinuationToken != "" {
		query.Set("continuation-token", input.ContinuationToken)
	}
	if input.StartAfter != "" {
		query.Set("start-after", input.StartAfter)
	}
	if input.MaxKeys > 0 {
		query.Set("max-keys", strconv.Itoa(input.MaxKeys))
	}

	res, err := client.do(ctx, http.MethodGet, input.Bucket, "", query, nil, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	var output ListObjectsOutput
	if err := xml.NewDecoder(res.Body).Decode(&output); err != nil {
		return nil, fmt.Errorf("s3: decoding list objects response: %w", err)
	}

	return &output, nil
}

func populateObjectHeaders(dst interface {
	setContentLength(int64)
	setContentType(string)
	setETag(string)
	setLastModified(time.Time)
	setMetadata(map[string]string)
	setVersionID(string)
}, headers http.Header) {
	if value := headers.Get("Content-Length"); value != "" {
		if parsed, err := strconv.ParseInt(value, 10, 64); err == nil {
			dst.setContentLength(parsed)
		}
	}
	dst.setContentType(headers.Get("Content-Type"))
	dst.setETag(headers.Get("ETag"))
	if value := headers.Get("Last-Modified"); value != "" {
		if parsed, err := http.ParseTime(value); err == nil {
			dst.setLastModified(parsed)
		}
	}
	dst.setVersionID(headers.Get("x-amz-version-id"))

	metadata := make(map[string]string)
	for key, value := range headers {
		lowerKey := strings.ToLower(key)
		if !strings.HasPrefix(lowerKey, metadataHeaderKey) || len(value) == 0 {
			continue
		}
		metadata[strings.TrimPrefix(lowerKey, metadataHeaderKey)] = value[0]
	}
	dst.setMetadata(metadata)
}

func validateBucketAndKey(bucket, key string) error {
	if bucket == "" {
		return errors.New("s3: bucket is required")
	}
	if key == "" {
		return errors.New("s3: key is required")
	}
	return nil
}

func inputBucketKey(input interface{}) (string, string) {
	switch value := input.(type) {
	case *GetObjectInput:
		return value.Bucket, value.Key
	case *HeadObjectInput:
		return value.Bucket, value.Key
	case *PutObjectInput:
		return value.Bucket, value.Key
	case *CreateMultipartUploadInput:
		return value.Bucket, value.Key
	case *UploadPartInput:
		return value.Bucket, value.Key
	case *CompleteMultipartUploadInput:
		return value.Bucket, value.Key
	case *AbortMultipartUploadInput:
		return value.Bucket, value.Key
	case *DeleteObjectInput:
		return value.Bucket, value.Key
	case *CopyObjectInput:
		return value.Bucket, value.Key
	default:
		return "", ""
	}
}

type objectIdentifier struct {
	Key       string `xml:"Key"`
	VersionID string `xml:"VersionId,omitempty"`
}

func toDeleteIdentifiers(input []ObjectIdentifier) []objectIdentifier {
	output := make([]objectIdentifier, 0, len(input))
	for _, item := range input {
		output = append(output, objectIdentifier{Key: item.Key, VersionID: item.VersionID})
	}
	return output
}

func (output *GetObjectOutput) setContentLength(value int64)    { output.ContentLength = value }
func (output *GetObjectOutput) setContentType(value string)     { output.ContentType = value }
func (output *GetObjectOutput) setETag(value string)            { output.ETag = value }
func (output *GetObjectOutput) setLastModified(value time.Time) { output.LastModified = value }
func (output *GetObjectOutput) setMetadata(value map[string]string) {
	output.Metadata = value
}
func (output *GetObjectOutput) setVersionID(value string) { output.VersionID = value }

func (output *HeadObjectOutput) setContentLength(value int64)    { output.ContentLength = value }
func (output *HeadObjectOutput) setContentType(value string)     { output.ContentType = value }
func (output *HeadObjectOutput) setETag(value string)            { output.ETag = value }
func (output *HeadObjectOutput) setLastModified(value time.Time) { output.LastModified = value }
func (output *HeadObjectOutput) setMetadata(value map[string]string) {
	output.Metadata = value
}
func (output *HeadObjectOutput) setVersionID(value string) { output.VersionID = value }
