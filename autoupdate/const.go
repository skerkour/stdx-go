package autoupdate

import "github.com/skerkour/stdx-go/crypto"

const (
	SaltSize = crypto.KeySize256

	ReleaseManifestFilename = "release.json"

	DefaultUserAgent = "Mozilla/5.0 (compatible; +autoupdate)"
)
