package s3

import (
	"bytes"
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type Bucket struct {
	Name         string    `xml:"Name"`
	CreationDate time.Time `xml:"CreationDate"`
}

type BucketOwner struct {
	ID          string `xml:"ID"`
	DisplayName string `xml:"DisplayName"`
}

type ListBucketsOutput struct {
	Owner   BucketOwner
	Buckets []Bucket
}

type HeadBucketInput struct {
	Bucket string
}

type HeadBucketOutput struct {
	Region string
}

type CreateBucketInput struct {
	Bucket string
}

type CreateBucketOutput struct {
	Location string
}

type DeleteBucketInput struct {
	Bucket string
}

type DeleteBucketOutput struct{}

type GetBucketLocationInput struct {
	Bucket string
}

type GetBucketLocationOutput struct {
	Region string
}

func (client *Client) ListBuckets(ctx context.Context) (*ListBucketsOutput, error) {
	res, err := client.do(ctx, http.MethodGet, "", "", nil, nil, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	var response struct {
		Owner   BucketOwner `xml:"Owner"`
		Buckets struct {
			Items []Bucket `xml:"Bucket"`
		} `xml:"Buckets"`
	}
	if err := xml.NewDecoder(res.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("s3: decoding list buckets response: %w", err)
	}

	return &ListBucketsOutput{
		Owner:   response.Owner,
		Buckets: response.Buckets.Items,
	}, nil
}

func (client *Client) HeadBucket(ctx context.Context, input *HeadBucketInput) (*HeadBucketOutput, error) {
	if input == nil {
		return nil, errors.New("s3: head bucket input is required")
	}
	if input.Bucket == "" {
		return nil, errors.New("s3: bucket is required")
	}

	res, err := client.do(ctx, http.MethodHead, input.Bucket, "", nil, nil, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	return &HeadBucketOutput{Region: res.Header.Get("x-amz-bucket-region")}, nil
}

func (client *Client) CreateBucket(ctx context.Context, input *CreateBucketInput) (*CreateBucketOutput, error) {
	if input == nil {
		return nil, errors.New("s3: create bucket input is required")
	}
	if input.Bucket == "" {
		return nil, errors.New("s3: bucket is required")
	}

	headers := http.Header{}
	var body []byte
	if client.region != defaultRegion {
		request, err := xml.Marshal(struct {
			XMLName            xml.Name `xml:"CreateBucketConfiguration"`
			LocationConstraint string   `xml:"LocationConstraint"`
		}{
			LocationConstraint: client.region,
		})
		if err != nil {
			return nil, fmt.Errorf("s3: encoding create bucket request: %w", err)
		}
		body = request
		headers.Set("Content-Type", "application/xml")
	}

	reader := bytes.NewReader(body)
	payloadHash := hashSHA256Hex(body)
	if len(body) == 0 {
		reader = bytes.NewReader(nil)
		payloadHash = emptyPayloadHash
	}

	res, err := client.do(ctx, http.MethodPut, input.Bucket, "", nil, headers, reader, int64(len(body)), payloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	location := strings.TrimSpace(res.Header.Get("Location"))
	return &CreateBucketOutput{Location: location}, nil
}

func (client *Client) DeleteBucket(ctx context.Context, input *DeleteBucketInput) (*DeleteBucketOutput, error) {
	if input == nil {
		return nil, errors.New("s3: delete bucket input is required")
	}
	if input.Bucket == "" {
		return nil, errors.New("s3: bucket is required")
	}

	res, err := client.do(ctx, http.MethodDelete, input.Bucket, "", nil, nil, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	return &DeleteBucketOutput{}, nil
}

func (client *Client) GetBucketLocation(ctx context.Context, input *GetBucketLocationInput) (*GetBucketLocationOutput, error) {
	if input == nil {
		return nil, errors.New("s3: get bucket location input is required")
	}
	if input.Bucket == "" {
		return nil, errors.New("s3: bucket is required")
	}

	query := url.Values{"location": {""}}
	res, err := client.do(ctx, http.MethodGet, input.Bucket, "", query, nil, nil, 0, emptyPayloadHash)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, decodeAPIError(res)
	}

	var response struct {
		XMLName xml.Name `xml:"LocationConstraint"`
		Value   string   `xml:",chardata"`
	}
	if err := xml.NewDecoder(res.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("s3: decoding get bucket location response: %w", err)
	}

	region := strings.TrimSpace(response.Value)
	if region == "" {
		region = defaultRegion
	}
	if region == "EU" {
		region = "eu-west-1"
	}

	return &GetBucketLocationOutput{Region: region}, nil
}
