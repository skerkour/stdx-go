package proto

// readVarint decodes a QUIC-style varint, as used for relay frame tags.
// It returns the value and the number of bytes consumed (0 on a short input).
func readVarint(b []byte) (uint64, int) {
	if len(b) == 0 {
		return 0, 0
	}
	tag := b[0]
	switch tag & 0xc0 {
	case 0x00:
		return uint64(tag), 1
	case 0x40:
		if len(b) < 2 {
			return 0, 0
		}
		return uint64(tag&0x3f)<<8 | uint64(b[1]), 2
	case 0x80:
		if len(b) < 4 {
			return 0, 0
		}
		return uint64(tag&0x3f)<<24 | uint64(b[1])<<16 | uint64(b[2])<<8 | uint64(b[3]), 4
	default:
		if len(b) < 8 {
			return 0, 0
		}
		return uint64(tag&0x3f)<<56 | uint64(b[1])<<48 | uint64(b[2])<<40 | uint64(b[3])<<32 |
			uint64(b[4])<<24 | uint64(b[5])<<16 | uint64(b[6])<<8 | uint64(b[7]), 8
	}
}

// appendVarint appends a QUIC-style varint encoding of v to dst.
func appendVarint(dst []byte, v uint64) []byte {
	switch {
	case v < 1<<6:
		return append(dst, byte(v))
	case v < 1<<14:
		return append(dst, byte(v>>8)|0x40, byte(v))
	case v < 1<<30:
		return append(dst, byte(v>>24)|0x80, byte(v>>16), byte(v>>8), byte(v))
	default:
		return append(dst, byte(v>>56)|0xc0, byte(v>>48), byte(v>>40), byte(v>>32),
			byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
	}
}
