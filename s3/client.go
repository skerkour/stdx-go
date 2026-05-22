package s3

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

const (
	defaultRegion     = "us-east-1"
	serviceName       = "s3"
	signingAlgorithm  = "AWS4-HMAC-SHA256"
	emptyPayloadHash  = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	metadataHeaderKey = "x-amz-meta-"
)

var MostUsedActions = []string{
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

type ClientConfig struct {
	Endpoint     string
	Region       string
	Credentials  CredentialsProvider
	HTTPClient   *http.Client
	UsePathStyle bool
}

type Client struct {
	endpoint     *url.URL
	region       string
	credentials  CredentialsProvider
	httpClient   *http.Client
	usePathStyle bool
	now          func() time.Time
}

type CredentialsProvider interface {
	Retrieve(ctx context.Context) (CredentialsValue, error)
}

type CredentialsValue struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

type staticCredentialsProvider struct {
	value CredentialsValue
}

func StaticCredentials(accessKeyID, secretAccessKey, sessionToken string) CredentialsProvider {
	return staticCredentialsProvider{
		value: CredentialsValue{
			AccessKeyID:     accessKeyID,
			SecretAccessKey: secretAccessKey,
			SessionToken:    sessionToken,
		},
	}
}

func (provider staticCredentialsProvider) Retrieve(ctx context.Context) (CredentialsValue, error) {
	if err := ctx.Err(); err != nil {
		return CredentialsValue{}, err
	}

	if provider.value.AccessKeyID == "" {
		return CredentialsValue{}, errors.New("s3: access key id is required")
	}
	if provider.value.SecretAccessKey == "" {
		return CredentialsValue{}, errors.New("s3: secret access key is required")
	}

	return provider.value, nil
}

type APIError struct {
	StatusCode int
	Code       string
	Message    string
	Resource   string
	RequestID  string
	HostID     string
}

func (err *APIError) Error() string {
	if err == nil {
		return ""
	}
	if err.Code != "" && err.Message != "" {
		return fmt.Sprintf("s3: %s: %s", err.Code, err.Message)
	}
	if err.Message != "" {
		return "s3: " + err.Message
	}

	return fmt.Sprintf("s3: unexpected status code %d", err.StatusCode)
}

func NewClient(config *ClientConfig) (*Client, error) {
	if config == nil {
		return nil, errors.New("s3: client config is required")
	}
	if config.Endpoint == "" {
		return nil, errors.New("s3: endpoint is required")
	}
	if config.Credentials == nil {
		return nil, errors.New("s3: credentials are required")
	}

	endpoint, err := url.Parse(config.Endpoint)
	if err != nil {
		return nil, fmt.Errorf("s3: parsing endpoint: %w", err)
	}
	if endpoint.Scheme != "http" && endpoint.Scheme != "https" {
		return nil, errors.New("s3: endpoint must use http or https")
	}
	if endpoint.Host == "" {
		return nil, errors.New("s3: endpoint host is required")
	}

	credentials, err := config.Credentials.Retrieve(context.Background())
	if err != nil {
		return nil, err
	}
	if credentials.AccessKeyID == "" {
		return nil, errors.New("s3: access key id is required")
	}
	if credentials.SecretAccessKey == "" {
		return nil, errors.New("s3: secret access key is required")
	}

	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = defaultHTTPClient()
	}

	region := config.Region
	if region == "" {
		region = defaultRegion
	}

	return &Client{
		endpoint:     endpoint,
		region:       region,
		credentials:  config.Credentials,
		httpClient:   httpClient,
		usePathStyle: config.UsePathStyle || shouldUsePathStyle(endpoint),
		now:          func() time.Time { return time.Now().UTC() },
	}, nil
}

func (client *Client) do(ctx context.Context, method, bucket, key string, query url.Values, headers http.Header, body io.Reader, contentLength int64, payloadHash string) (*http.Response, error) {
	requestURL, canonicalURI, canonicalQuery, err := client.buildURL(bucket, key, query)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, method, requestURL, body)
	if err != nil {
		return nil, fmt.Errorf("s3: building request: %w", err)
	}
	if headers != nil {
		req.Header = headers.Clone()
	}
	if body == nil {
		req.Body = nil
	}
	if contentLength > 0 {
		req.ContentLength = contentLength
	}

	if err := client.signRequest(ctx, req, canonicalURI, canonicalQuery, payloadHash); err != nil {
		return nil, err
	}

	res, err := client.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("s3: sending request: %w", err)
	}

	return res, nil
}

func (client *Client) buildURL(bucket, key string, query url.Values) (requestURL string, canonicalURI string, canonicalQuery string, err error) {
	host := client.endpoint.Host
	basePath := strings.TrimRight(client.endpoint.EscapedPath(), "/")

	objectPath := "/"
	if key != "" {
		objectPath = "/" + uriEncode(strings.TrimPrefix(key, "/"), false)
	}

	if client.usePathStyle {
		if bucket == "" {
			canonicalURI = basePath + objectPath
		} else {
			canonicalURI = basePath + "/" + uriEncode(strings.TrimPrefix(bucket, "/"), true)
			if key != "" {
				canonicalURI += "/" + uriEncode(strings.TrimPrefix(key, "/"), false)
			}
		}
	} else {
		canonicalURI = basePath + objectPath
		if bucket != "" {
			host = bucket + "." + host
		}
	}

	if canonicalURI == "" {
		canonicalURI = "/"
	}

	canonicalQuery = canonicalQueryString(query)
	requestURL = client.endpoint.Scheme + "://" + host + canonicalURI
	if canonicalQuery != "" {
		requestURL += "?" + canonicalQuery
	}

	return requestURL, canonicalURI, canonicalQuery, nil
}

func (client *Client) signRequest(ctx context.Context, req *http.Request, canonicalURI, canonicalQuery, payloadHash string) error {
	credentials, err := client.credentials.Retrieve(ctx)
	if err != nil {
		return err
	}

	now := client.now().UTC()
	amzDate := now.Format("20060102T150405Z")
	shortDate := now.Format("20060102")

	req.Header.Set("Host", req.URL.Host)
	req.Header.Set("x-amz-date", amzDate)
	req.Header.Set("x-amz-content-sha256", payloadHash)
	if credentials.SessionToken != "" {
		req.Header.Set("x-amz-security-token", credentials.SessionToken)
	}

	canonicalHeaders, signedHeaders := canonicalHeaders(req.Header, req.URL.Host)
	scope := shortDate + "/" + client.region + "/" + serviceName + "/aws4_request"
	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQuery,
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	stringToSign := strings.Join([]string{
		signingAlgorithm,
		amzDate,
		scope,
		hashSHA256Hex([]byte(canonicalRequest)),
	}, "\n")

	signature := hex.EncodeToString(hmacSHA256(deriveSigningKey(credentials.SecretAccessKey, shortDate, client.region, serviceName), stringToSign))
	req.Header.Set("Authorization", signingAlgorithm+" Credential="+credentials.AccessKeyID+"/"+scope+", SignedHeaders="+signedHeaders+", Signature="+signature)
	return nil
}

func canonicalHeaders(headers http.Header, host string) (string, string) {
	values := map[string]string{
		"host": host,
	}

	for key, value := range headers {
		lowerKey := strings.ToLower(strings.TrimSpace(key))
		if lowerKey == "authorization" {
			continue
		}

		normalizedValues := make([]string, 0, len(value))
		for _, item := range value {
			normalizedValues = append(normalizedValues, strings.Join(strings.Fields(item), " "))
		}
		values[lowerKey] = strings.Join(normalizedValues, ",")
	}

	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var headerBuilder strings.Builder
	for _, key := range keys {
		headerBuilder.WriteString(key)
		headerBuilder.WriteByte(':')
		headerBuilder.WriteString(values[key])
		headerBuilder.WriteByte('\n')
	}

	return headerBuilder.String(), strings.Join(keys, ";")
}

func canonicalQueryString(values url.Values) string {
	if len(values) == 0 {
		return ""
	}

	type pair struct {
		key   string
		value string
	}

	pairs := make([]pair, 0, len(values))
	for key, list := range values {
		encodedKey := uriEncode(key, true)
		if len(list) == 0 {
			pairs = append(pairs, pair{key: encodedKey})
			continue
		}
		for _, value := range list {
			pairs = append(pairs, pair{
				key:   encodedKey,
				value: uriEncode(value, true),
			})
		}
	}

	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].key == pairs[j].key {
			return pairs[i].value < pairs[j].value
		}
		return pairs[i].key < pairs[j].key
	})

	var builder strings.Builder
	for index, pair := range pairs {
		if index > 0 {
			builder.WriteByte('&')
		}
		builder.WriteString(pair.key)
		builder.WriteByte('=')
		builder.WriteString(pair.value)
	}

	return builder.String()
}

func preparePayload(body io.Reader, payloadHash string, contentLength int64) (io.Reader, string, int64, error) {
	if payloadHash != "" {
		return body, payloadHash, contentLength, nil
	}

	if body == nil {
		return nil, emptyPayloadHash, 0, nil
	}

	if seeker, ok := body.(io.ReadSeeker); ok {
		hash := sha256.New()
		if _, err := seeker.Seek(0, io.SeekStart); err != nil {
			return nil, "", 0, fmt.Errorf("s3: seeking request body: %w", err)
		}
		if _, err := io.Copy(hash, seeker); err != nil {
			return nil, "", 0, fmt.Errorf("s3: hashing request body: %w", err)
		}
		if _, err := seeker.Seek(0, io.SeekStart); err != nil {
			return nil, "", 0, fmt.Errorf("s3: rewinding request body: %w", err)
		}

		if contentLength <= 0 {
			size, err := bodyLength(seeker)
			if err != nil {
				return nil, "", 0, err
			}
			contentLength = size
		}

		return seeker, hex.EncodeToString(hash.Sum(nil)), contentLength, nil
	}

	data, err := io.ReadAll(body)
	if err != nil {
		return nil, "", 0, fmt.Errorf("s3: reading request body: %w", err)
	}
	if contentLength <= 0 {
		contentLength = int64(len(data))
	}

	sum := sha256.Sum256(data)
	return bytes.NewReader(data), hex.EncodeToString(sum[:]), contentLength, nil
}

func bodyLength(seeker io.ReadSeeker) (int64, error) {
	current, err := seeker.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, fmt.Errorf("s3: seeking request body: %w", err)
	}
	end, err := seeker.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, fmt.Errorf("s3: seeking request body: %w", err)
	}
	if _, err := seeker.Seek(current, io.SeekStart); err != nil {
		return 0, fmt.Errorf("s3: seeking request body: %w", err)
	}

	return end - current, nil
}

func decodeAPIError(res *http.Response) error {
	defer res.Body.Close()

	payload, err := io.ReadAll(res.Body)
	if err != nil {
		return fmt.Errorf("s3: reading error response: %w", err)
	}

	apiErr := &APIError{StatusCode: res.StatusCode}
	if len(payload) == 0 {
		return apiErr
	}

	var xmlErr struct {
		Code      string `xml:"Code"`
		Message   string `xml:"Message"`
		Resource  string `xml:"Resource"`
		RequestID string `xml:"RequestId"`
		HostID    string `xml:"HostId"`
	}
	if err := xml.Unmarshal(payload, &xmlErr); err == nil {
		apiErr.Code = xmlErr.Code
		apiErr.Message = xmlErr.Message
		apiErr.Resource = xmlErr.Resource
		apiErr.RequestID = xmlErr.RequestID
		apiErr.HostID = xmlErr.HostID
		return apiErr
	}

	apiErr.Message = strings.TrimSpace(string(payload))
	return apiErr
}

func deriveSigningKey(secretKey, shortDate, region, service string) []byte {
	dateKey := hmacSHA256([]byte("AWS4"+secretKey), shortDate)
	regionKey := hmacSHA256(dateKey, region)
	serviceKey := hmacSHA256(regionKey, service)
	return hmacSHA256(serviceKey, "aws4_request")
}

func hmacSHA256(key []byte, data string) []byte {
	hash := hmac.New(sha256.New, key)
	hash.Write([]byte(data))
	return hash.Sum(nil)
}

func hashSHA256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func shouldUsePathStyle(endpoint *url.URL) bool {
	host := endpoint.Hostname()
	if host == "localhost" {
		return true
	}
	if ip := net.ParseIP(host); ip != nil {
		return true
	}
	return endpoint.Port() != ""
}

func uriEncode(value string, encodeSlash bool) string {
	if value == "" {
		return ""
	}

	var builder strings.Builder
	for index := 0; index < len(value); index++ {
		character := value[index]
		if isUnreserved(character) || (!encodeSlash && character == '/') {
			builder.WriteByte(character)
			continue
		}
		builder.WriteString(fmt.Sprintf("%%%02X", character))
	}

	return builder.String()
}

func isUnreserved(character byte) bool {
	switch {
	case character >= 'A' && character <= 'Z':
		return true
	case character >= 'a' && character <= 'z':
		return true
	case character >= '0' && character <= '9':
		return true
	case character == '-', character == '.', character == '_', character == '~':
		return true
	default:
		return false
	}
}

func defaultHTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 60 * time.Second,
			}).DialContext,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: time.Second,
		},
	}
}
