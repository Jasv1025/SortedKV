/*
Wraps POSIX mmap/munmap (or a platform abstraction) for the sorted key file.
Provides:
    class MmapKeyList {
    public:
        uint64_t size() const;
        uint64_t operator[](uint64_t idx) const;
        // iterators, etc.
    };
*/