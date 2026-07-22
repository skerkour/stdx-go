# postgresql-backup

Encrypted PostgreSQL backups to S3-compatible storage.

- `pg_dump` -> gzip -> XChaCha20-Poly1305 encrypt -> S3 upload
- X-Wing hybrid KEM (X25519 + ML-KEM-768) + HKDF-SHA-512 key exchange per backup
- Cron-based scheduling

## Usage

**Generate an X-Wing keypair:**
```
postgresql-backup -generate
```

**Run the backup daemon:**
```
postgresql-backup -config config.yaml
```

**Decrypt a backup:**
```
KEY=<xwing-secret-key-base64> postgresql-backup -decrypt backup.sql.gz.enc -out restore.sql
# auto-derives output path (strips .enc):
KEY=... postgresql-backup -decrypt backup.sql.gz.enc
```

**Restore a decrypted backup:**
```
psql "$DATABASE_URL" < restore.sql
```

## Docker

```bash
$ docker run -d -v `pwd`/config.yml:/app/config.yml:ro ghcr.io/skerkour/postgresql-backup
```

## Config

```yaml
s3:
  endpoint: https://s3.example.com
  bucket: my-backups
  access_key_id: AKIAIOSFODNN7EXAMPLE
  secret_access_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

databases:
  - url: postgres://user:pass@host:5432/mydb
    public_key: <xwing-encapsulation-key-base64>   # 1624 base64 chars (1216-byte X-Wing encapsulation key)
    cron: "0 0 * * *"                              # 5-field cron
    folder: myapp-prod                             # S3 key prefix
```

## Encrypted file format

```
[1120 bytes: xwing ciphertext]
[24 bytes: random nonce]
[ciphertext + XChaCha20-Poly1305 auth tag]
```

## S3 key format

```
<folder>/<RFC3339 UTC>.sql.gz.enc
```
