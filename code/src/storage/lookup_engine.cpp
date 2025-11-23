/*
Implements:
    std::optional<uint64_t> lookup(uint64_t key);
    std::vector<std::optional<uint64_t>> batch_lookup(span<const uint64_t> keys);
Uses:
    Global model → leaf index
    Leaf model → predicted position
    search.cpp for error-window search.
*/