#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdexcept>
#include <memory>
#include <functional>

template <typename T>
T read_pod(std::string_view view, size_t offset) {
    if (offset + sizeof(T) > view.size()) {
        throw std::out_of_range("Buffer overflow reading POD");
    }
    T val;
    std::memcpy(&val, view.data() + offset, sizeof(T));
    return val;
}

struct KV {
    std::vector<uint8_t> k;
    std::vector<uint8_t> v;
    
    bool operator<(const KV& o) const {
        return k < o.k;
    }
};

class BlockLSMWriter {
private:
    static void flush_block(std::vector<KV>& batch, FILE* f, size_t block_size, size_t& current_overhead) {
        if (batch.empty()) {
            return;
        }
        
        uint16_t count = static_cast<uint16_t>(batch.size());
        uint16_t total_key_bytes = 0;
        
        for (const auto& b : batch) {
            total_key_bytes += static_cast<uint16_t>(b.k.size());
        }

        fwrite(&count, sizeof(uint16_t), 1, f);
        fwrite(&total_key_bytes, sizeof(uint16_t), 1, f);

        std::vector<uint16_t> k_lens;
        std::vector<uint16_t> v_lens;
        
        k_lens.reserve(count);
        v_lens.reserve(count);

        for (const auto& b : batch) {
            k_lens.push_back(static_cast<uint16_t>(b.k.size()));
        }
        for (const auto& b : batch) {
            v_lens.push_back(static_cast<uint16_t>(b.v.size()));
        }

        fwrite(k_lens.data(), sizeof(uint16_t), count, f);
        fwrite(v_lens.data(), sizeof(uint16_t), count, f);

        for (const auto& b : batch) {
            fwrite(b.k.data(), 1, b.k.size(), f);
        }
        for (const auto& b : batch) {
            fwrite(b.v.data(), 1, b.v.size(), f);
        }

        size_t written = 4 + (count * 4) + total_key_bytes;
        for (const auto& b : batch) {
            written += b.v.size();
        }

        if (written < block_size) {
            std::vector<uint8_t> pad(block_size - written, 0);
            fwrite(pad.data(), 1, pad.size(), f);
        }
        
        batch.clear();
        current_overhead = 4;
    }

public:
    static void write(const std::string& filename, 
                      size_t block_size,
                      const std::vector<std::vector<uint8_t>>& keys, 
                      const std::vector<std::vector<uint8_t>>& values) {
        
        if (keys.size() != values.size()) {
            throw std::runtime_error("Size mismatch between keys and values");
        }

        std::vector<KV> data(keys.size());
        for (size_t i = 0; i < keys.size(); i++) {
            data[i] = {keys[i], values[i]};
        }
        
        std::sort(data.begin(), data.end());

        std::unique_ptr<FILE, decltype(&fclose)> f(fopen(filename.c_str(), "wb"), &fclose);
        if (!f) {
            throw std::runtime_error("Failed to open file for writing");
        }

        std::vector<KV> batch;
        size_t current_overhead = 4;

        for (const auto& item : data) {
            size_t item_sz = 4 + item.k.size() + item.v.size();
            
            if (current_overhead + item_sz > block_size) {
                if (batch.empty()) {
                    throw std::runtime_error("Single item is too large for the configured block size");
                }
                flush_block(batch, f.get(), block_size, current_overhead);
            }
            
            batch.push_back(item);
            current_overhead += item_sz;
        }

        if (!batch.empty()) {
            flush_block(batch, f.get(), block_size, current_overhead);
        }
    }
};

class MmapBlockReader {
private:
    int fd;
    size_t file_sz;
    size_t block_sz;
    
    std::unique_ptr<char, std::function<void(char*)>> mmap_ptr;

public:
    MmapBlockReader(const std::string& filename, size_t block_size) 
        : block_sz(block_size), mmap_ptr(nullptr, [](char*){}) {
            
        fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Open failed");
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Stat failed");
        }
        file_sz = static_cast<size_t>(sb.st_size);

        void* raw_ptr = mmap(nullptr, file_sz, PROT_READ, MAP_SHARED, fd, 0);
        if (raw_ptr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Mmap failed");
        }

        size_t captured_size = file_sz;
        mmap_ptr = std::unique_ptr<char, std::function<void(char*)>>(
            static_cast<char*>(raw_ptr),
            [captured_size](char* ptr) {
                if (ptr) {
                    munmap(ptr, captured_size);
                }
            }
        );
    }

    ~MmapBlockReader() {
        if (fd != -1) {
            close(fd);
        }
    }

    size_t num_blocks() const {
        return file_sz / block_sz;
    }

    class BlockView {
    private:
        std::string_view view;
        uint16_t count;
        uint16_t key_blob_sz;

        size_t k_lens_offset;
        size_t v_lens_offset;
        size_t k_blob_offset;
        size_t v_blob_offset;

    public:
        BlockView(std::string_view block_data) : view(block_data) {
            count = read_pod<uint16_t>(view, 0);
            key_blob_sz = read_pod<uint16_t>(view, 2);

            k_lens_offset = 4;
            v_lens_offset = 4 + (static_cast<size_t>(count) * 2);
            k_blob_offset = v_lens_offset + (static_cast<size_t>(count) * 2);
            v_blob_offset = k_blob_offset + key_blob_sz;
        }

        uint16_t size() const { 
            return count; 
        }

        uint64_t get_key_int(uint16_t idx) const {
            if (idx >= count) {
                return 0;
            }
            
            uint32_t blob_offset = 0;
            for (int i = 0; i < idx; i++) {
                blob_offset += read_pod<uint16_t>(view, k_lens_offset + (i * 2));
            }
            
            uint16_t len = read_pod<uint16_t>(view, k_lens_offset + (idx * 2));
            
            std::string_view key_view = view.substr(k_blob_offset + blob_offset, len);
            
            uint64_t k = 0;
            size_t copy_len = std::min(static_cast<size_t>(len), sizeof(uint64_t));
            std::memcpy(&k, key_view.data(), copy_len);
            
            return k;
        }

        std::string get_val_str(uint16_t idx) const {
            if (idx >= count) {
                return "";
            }

            uint32_t blob_offset = 0;
            for (int i = 0; i < idx; i++) {
                blob_offset += read_pod<uint16_t>(view, v_lens_offset + (i * 2));
            }
            
            uint16_t len = read_pod<uint16_t>(view, v_lens_offset + (idx * 2));

            return std::string(view.substr(v_blob_offset + blob_offset, len));
        }
    };

    BlockView get_block(size_t block_idx) const {
        size_t offset = block_idx * block_sz;
        if (offset >= file_sz) {
            throw std::out_of_range("Block index out of bounds");
        }
        
        std::string_view sv(mmap_ptr.get() + offset, block_sz);
        return BlockView(sv);
    }
};