# s3

Minimal, stdlib-only S3 client.

## Create a client

```go
package main

import (
"context"
"log"

"github.com/skerkour/stdx-go/s3"
)

func main() {
client, err := s3.NewClient(&s3.ClientConfig{
Endpoint: "https://s3.amazonaws.com",
Region:   "us-east-1",
Credentials: s3.StaticCredentials(
"ACCESS_KEY_ID",
"SECRET_ACCESS_KEY",
"SESSION_TOKEN", // optional
),
})
if err != nil {
log.Fatal(err)
}

ctx := context.Background()
_ = ctx
_ = client
}
```

## Object operations

```go
ctx := context.Background()

// PutObject
_, err := client.PutObject(ctx, &s3.PutObjectInput{
Bucket:      "my-bucket",
Key:         "docs/readme.txt",
Body:        strings.NewReader("hello"),
ContentType: "text/plain",
})
if err != nil {
log.Fatal(err)
}

// GetObject
getOut, err := client.GetObject(ctx, &s3.GetObjectInput{
Bucket: "my-bucket",
Key:    "docs/readme.txt",
})
if err != nil {
log.Fatal(err)
}
defer getOut.Body.Close()

// DeleteObjects
_, err = client.DeleteObjects(ctx, &s3.DeleteObjectsInput{
Bucket: "my-bucket",
Objects: []s3.ObjectIdentifier{
{Key: "docs/readme.txt"},
{Key: "docs/old.txt"},
},
})
if err != nil {
log.Fatal(err)
}
```

## Bucket operations

```go
ctx := context.Background()

// ListBuckets
listBucketsOut, err := client.ListBuckets(ctx)
if err != nil {
log.Fatal(err)
}
_ = listBucketsOut

// HeadBucket
headOut, err := client.HeadBucket(ctx, &s3.HeadBucketInput{Bucket: "my-bucket"})
if err != nil {
log.Fatal(err)
}
log.Println("bucket region:", headOut.Region)

// GetBucketLocation
locationOut, err := client.GetBucketLocation(ctx, &s3.GetBucketLocationInput{Bucket: "my-bucket"})
if err != nil {
log.Fatal(err)
}
log.Println("location:", locationOut.Region)
```
