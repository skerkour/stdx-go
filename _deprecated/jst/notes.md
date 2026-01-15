use _ as a separator to be selectable?
- conflict with base64 URL cahcaracter set

should be easy to find by regexp

e.g.

jst_v1_local_[header]_[payload]

header:
- expires_at
- not_before
- issued_at
- token_id: to prevent replay attacks
- key_id
- nonce
