# TODO: Move to sdk

class Uint256:
    def __init__(self, value=0):
        if isinstance(value, list) and len(value) == 4:
            # Initialize from list of four 64-bit integers
            self.value = sum(v << (64 * i) for i, v in enumerate(value))
        elif isinstance(value, bytes):
            self.value = int.from_bytes(value, "little")
        else:
            self.value = value & ((1 << 256) - 1)  # Ensure 256-bit value

    @classmethod
    def from_compact_target_bits(cls, bits):
        # Extract mantissa and exponent
        unshifted_expt = bits >> 24
        if unshifted_expt <= 3:
            mant = (bits & 0xFFFFFF) >> (8 * (3 - unshifted_expt))
            expt = 0
        else:
            mant = bits & 0xFFFFFF
            expt = 8 * (unshifted_expt - 3)

        # Check if mantissa is valid
        if mant > 0x7FFFFF:
            return cls(0)  # Return Uint256::ZERO equivalent
        else:
            # Shift mantissa by exponent
            result = mant << expt
            return cls(result)

    def __lshift__(self, shift):
        return Uint256((self.value << shift) & ((1 << 256) - 1))

    def __str__(self):
        return f"Uint256({self.value})"

    def __repr__(self):
        return self.__str__()
    
    def hex(self):
        return f"{self.value:064x}"
    
    def __eq__(self, other):
        if isinstance(other, Uint256):
            return self.value == other.value
        return self.value == other

    def __lt__(self, other):
        if isinstance(other, Uint256):
            return self.value < other.value
        return self.value < other

    def __gt__(self, other):
        if isinstance(other, Uint256):
            return self.value > other.value
        return self.value > other

    def __le__(self, other):
        if isinstance(other, Uint256):
            return self.value <= other.value
        return self.value <= other

    def __ge__(self, other):
        if isinstance(other, Uint256):
            return self.value >= other.value
        return self.value >= other
    
class XoShiRo256PlusPlusLimb:
    def __init__(self, value):
        self.value = value & 0xFFFFFFFFFFFFFFFF

    def __add__(self, other):
        return XoShiRo256PlusPlusLimb((self.value + other.value) & 0xFFFFFFFFFFFFFFFF)

    def __xor__(self, other):
        return XoShiRo256PlusPlusLimb(self.value ^ other.value)

    def __lshift__(self, shift):
        return XoShiRo256PlusPlusLimb((self.value << shift) & 0xFFFFFFFFFFFFFFFF)

    def rotate_left(self, shift):
        return XoShiRo256PlusPlusLimb(((self.value << shift) | (self.value >> (64 - shift))) & 0xFFFFFFFFFFFFFFFF)

class Hash:
    def __init__(self, data):
        if isinstance(data, bytes) and len(data) == 32:
            self.data = data
        else:
            raise ValueError("Hash must be initialized with 32 bytes")

    def to_le_u64(self):
        return [int.from_bytes(self.data[i:i+8], 'little') for i in range(0, 32, 8)]

class XoShiRo256PlusPlus:
    def __init__(self, hash):
        hash_u64 = hash.to_le_u64()
        self.s0 = XoShiRo256PlusPlusLimb(hash_u64[0])
        self.s1 = XoShiRo256PlusPlusLimb(hash_u64[1])
        self.s2 = XoShiRo256PlusPlusLimb(hash_u64[2])
        self.s3 = XoShiRo256PlusPlusLimb(hash_u64[3])

    def u64(self):
        res = self.s0 + (self.s0 + self.s3).rotate_left(23)
        t = self.s1 << 17
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3

        self.s2 ^= t
        self.s3 = self.s3.rotate_left(45)

        return res.value