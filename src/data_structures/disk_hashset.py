# Disk-backed fixed-slot hashset for ASCII/UTF-8 strings up to 20 bytes.
# Uses open addressing with linear probing and mmap.

import mmap
import os
import struct

HEADER_SIZE = 4096
MAGIC = b"DHSETv1\x00"
VERSION = 1
SLOT_SIZE = 32  # bytes
MAX_KEY_BYTES = 20

# slot format: flag(1), key_len(1), fingerprint(8), key_bytes(20), pad(2)
SLOT_STRUCT = struct.Struct("<B B Q 20s 2s")

FLAG_EMPTY = 0
FLAG_OCCUPIED = 1
FLAG_TOMBSTONE = 2


# From: https://gist.github.com/ruby0x1/81308642d0325fd386237cfa3b44785c#file-hash_fnv1a-h-L25
def fnv1a_64(data: bytes) -> int:
    # 64-bit FNV-1a
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF  # 64-bit overflow
    return h


class DiskHashSet:
    def __init__(self, path: str, num_slots_power: int = 20, create: bool = False):
        """
        path: file path
        num_slots_power: use 2**num_slots_power slots (must be >= 3)
        create: if True, create new file (overwrite)
        """
        assert num_slots_power >= 3
        self.path = path
        self.num_slots = 1 << num_slots_power
        self.capacity_mask = self.num_slots - 1
        self.slot_size = SLOT_SIZE

        exists = os.path.exists(path)
        mode = "r+b"
        if create or not exists:
            # create/truncate file with header + space for slots
            fsize = HEADER_SIZE + self.num_slots * self.slot_size
            with open(path, "wb") as f:
                f.truncate(fsize)
                header = bytearray(HEADER_SIZE)
                header[0:8] = MAGIC
                struct.pack_into("<I", header, 8, VERSION)
                struct.pack_into("<Q", header, 12, self.num_slots)
                struct.pack_into("<I", header, 20, self.slot_size)
                f.write(header)
            mode = "r+b"

        self.f = open(path, mode)
        self.mm = mmap.mmap(self.f.fileno(), 0)

        # sanity check header
        magic = self.mm[0:8]
        if magic != MAGIC:
            raise ValueError("Bad file magic; wrong file or version")
        # read num_slots from file; allow opening a file with different num_slots_power
        on_disk_slots = struct.unpack_from("<Q", self.mm, 12)[0]
        if on_disk_slots != self.num_slots:
            # allow opening with the file's native size, override
            self.num_slots = int(on_disk_slots)
            self.capacity_mask = self.num_slots - 1

        self.slots_base = HEADER_SIZE

    def _slot_offset(self, idx: int) -> int:
        return self.slots_base + (idx & self.capacity_mask) * self.slot_size

    def _read_slot(self, idx: int):
        off = self._slot_offset(idx)
        data = self.mm[off : off + self.slot_size]
        return SLOT_STRUCT.unpack(data)

    def _write_slot(
        self, idx: int, flag: int, key_len: int, fingerprint: int, key_bytes: bytes
    ):
        off = self._slot_offset(idx)
        kb = key_bytes.ljust(MAX_KEY_BYTES, b"\0")[:MAX_KEY_BYTES]
        packed = SLOT_STRUCT.pack(flag, key_len, fingerprint, kb, b"\0\0")
        self.mm[off : off + self.slot_size] = packed
        # optionally flush here for persistence (costly). Caller can call sync().

    def sync(self):
        self.mm.flush()

    def close(self):
        self.sync()
        self.mm.close()
        self.f.close()

    def _encode_key(self, key: str) -> bytes:
        b = key.encode("utf-8")
        if len(b) > MAX_KEY_BYTES:
            raise ValueError(f"key too long in bytes (max {MAX_KEY_BYTES}): {key!r}")
        return b

    def contains(self, key: str) -> bool:
        kb = self._encode_key(key)
        fp = fnv1a_64(kb)
        start = idx = fp & self.capacity_mask
        while True:
            flag, key_len, slot_fp, slot_kb, _ = self._read_slot(idx)
            if flag == FLAG_EMPTY:
                return False
            if flag == FLAG_OCCUPIED and slot_fp == fp:
                # compare exact bytes
                if slot_kb[:key_len] == kb:
                    return True
            idx = (idx + 1) & self.capacity_mask
            if idx == start:
                # table full and no match
                return False

    def add(self, key: str) -> bool:
        """
        Insert key. Return True if inserted, False if already present.
        """
        kb = self._encode_key(key)
        fp = fnv1a_64(kb)
        start = idx = fp & self.capacity_mask

        while True:
            flag, key_len, slot_fp, slot_kb, _ = self._read_slot(idx)
            if flag == FLAG_OCCUPIED:
                if slot_fp == fp and slot_kb[:key_len] == kb:
                    return False  # already present
            else:  # tombstone or empty
                self._write_slot(idx, FLAG_OCCUPIED, len(kb), fp, kb)
                return True
            idx = (idx + 1) & self.capacity_mask
            if idx == start:
                raise RuntimeError("Hash table is full; need resize")

    def remove(self, key: str) -> bool:
        kb = self._encode_key(key)
        fp = fnv1a_64(kb)
        start = idx = fp & self.capacity_mask
        while True:
            flag, key_len, slot_fp, slot_kb, _ = self._read_slot(idx)
            if flag == FLAG_EMPTY:
                return False
            if flag == FLAG_OCCUPIED and slot_fp == fp and slot_kb[:key_len] == kb:
                off = self._slot_offset(idx)
                # reconstruct slot with tombstone flag but preserve fingerprint+key_len/key
                packed = SLOT_STRUCT.pack(
                    FLAG_TOMBSTONE, key_len, slot_fp, slot_kb, b"\0\0"
                )
                self.mm[off : off + self.slot_size] = packed
                return True
            idx = (idx + 1) & self.capacity_mask
            if idx == start:
                return False

    def bulk_load(self, iterable):
        # naive: call add for each. For huge imports consider external sort by bucket and sequentially write.
        for k in iterable:
            self.add(k)


if __name__ == "__main__":
    path = "users_hashset.dat"
    # create new with 2**30 slots (~1.07B slots). choose power based on expected N and load factor.
    # For 1B keys you might pick 2**30 (1,073,741,824 slots) for each slot 32 bytes = 32GB file.
    hs = DiskHashSet(path, num_slots_power=30, create=True)

    print(hs.add("alice"))  # True
    print(hs.add("alice"))  # False
    print(hs.contains("alice"))  # True
    print(hs.remove("alice"))  # True
    print(hs.contains("alice"))  # False

    hs.close()
