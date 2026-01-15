# JSON Secure Token (JST)

```
JST_SECRET := base64(crypto.randBytes(64))
secret := base64Decode(JST_SECRET)
encryptionKey := secret[:32]
auhtKey := secret[32:]

prefix := "jst.v1.local."

header.nonce := crypto.randBytes(24)
encodedHeader := Base64Url(JSON(header))

encryptedPayload := XChaCha20-Poly1305.encrypt(key=encryptionKey, data=JSON(payload), nonce=nonce)
encodedPayload := Base64Url(encryptedPayload)

signature := HMAC-SHA-256(auhtKey, prefix || encodedHeader || "." || encodedPayload)
encodedSignature := Base64Url(signature)

Token := prefix || encodedHeader || "." || encodedPayload || "." || encodedSignature
```


```
encryptionContext := "jst-v1 2023-12-31 23:59:59:999 encryption-key"
authenticationContext := "jst-v1 2024-01-01 00:00:00:000 authentication-key"

jst_master_key := crypto.randBytes(32)

nonce := crypto.randBytes(24)
encryptionKey := BLAKE3.deriveKey(encryptionContext, jst_master_key)
authenticationKey := BLAKE3.deriveKey(authenticationContext, nonce || jst_master_key)

encryptedPayload := XChaCha20.encrypt(encryptionKey, nonce, payload)

signature := BLAKE3.keyed(authenticationKey, [TODO])

tokenSignature := extractSignature(Token)
signature := HMAC-SHA-256(auhtKey, prefix || encodedHeader || "." || encodedPayload)

if constantTimeCompare(tokenSignature, signature) == false {
    return error;
}

header := Base64UrlDecode(encodedHeader)

encryptedPayload := base64UrlDecode(encodedPayload)
decryptedPayload := XChaCha20-Poly1305.decrypt(key=encryptionKey, data=encryptedPayload, nonce=header.nonce)
```
