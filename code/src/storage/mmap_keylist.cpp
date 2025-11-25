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
#include <variant>

enum class KeyType {
    Bytes,
    Integer
};
struct Blob {
    std::vector<uint8_t> k;
    std::vector<uint8_t> v;
};

class BlockLSMWriter {
private:
    static void flush_block(std::vector<Blob>& batch, FILE* f, size_t block_size, size_t& current_overhead) {
        if (batch.empty()) {
            return;
        }
        
        uint16_t count = static_cast<uint16_t>(batch.size());

        std::vector<uint16_t> k_off;
        std::vector<uint16_t> v_off;
        k_off.reserve(count + 1);
        v_off.reserve(count + 1);

        k_off.push_back(0);
        v_off.push_back(0);

        uint16_t run_k = 0;
        uint16_t run_v = 0;

        for (const auto& b : batch) {
            run_k += static_cast<uint16_t>(b.k.size());
            k_off.push_back(run_k);

            run_v += static_cast<uint16_t>(b.v.size());
            v_off.push_back(run_v);
        }
        
        uint16_t total_key_bytes = run_k;

  
        fwrite(&count, sizeof(uint16_t), 1, f);
        fwrite(&total_key_bytes, sizeof(uint16_t), 1, f);


        fwrite(k_off.data(), sizeof(uint16_t), k_off.size(), f);
        fwrite(v_off.data(), sizeof(uint16_t), v_off.size(), f);

        for (const auto& b : batch) {
            fwrite(b.k.data(), 1, b.k.size(), f);
        }
        for (const auto& b : batch) {
            fwrite(b.v.data(), 1, b.v.size(), f);
        }

        // Calculate Padding to fill the rest of the block
        // Header (4) + K_Offsets + V_Offsets + K_Data + V_Data
        size_t headers_sz = 4 + (k_off.size() * 2) + (v_off.size() * 2);
        size_t data_sz = total_key_bytes + run_v;
        size_t written = headers_sz + data_sz;

        if (written < block_size) {
            std::vector<uint8_t> pad(block_size - written, 0);
            fwrite(pad.data(), 1, pad.size(), f);
        }
        
        batch.clear();
        current_overhead = 8;
    }

public:
    static void write(const std::string& filename, 
                      size_t block_size,
                      const std::vector<std::vector<uint8_t>>& keys, 
                      const std::vector<std::vector<uint8_t>>& values,
                      KeyType type = KeyType::Bytes) {
        
        if (keys.size() != values.size()) {
            throw std::runtime_error("Size mismatch between keys and values");
        }

        std::vector<Blob> data(keys.size());
        for (size_t i = 0; i < keys.size(); i++) {
            data[i] = {keys[i], values[i]};
        }

        if (type == KeyType::Integer) {
            std::sort(data.begin(), data.end(), [](const Blob& a, const Blob& b) {
                if (a.k.size() != 8 || b.k.size() != 8) {
                     return a.k < b.k;
                }
                uint64_t va;
                uint64_t vb;
                std::memcpy(&va, a.k.data(), 8);
                std::memcpy(&vb, b.k.data(), 8);
                return va < vb;
            });
        } else {
            std::sort(data.begin(), data.end(), [](const Blob& a, const Blob& b) {
                return a.k < b.k;
            });
        }

        std::unique_ptr<FILE, decltype(&fclose)> f(fopen(filename.c_str(), "wb"), &fclose);
        if (!f) {
            throw std::runtime_error("Failed to open file for writing");
        }

        std::vector<Blob> batch;
        size_t current_overhead = 8;

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
        
        void* ptr = mmap(nullptr, file_sz, PROT_READ, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) { 
            close(fd); 
            throw std::runtime_error("Mmap failed"); 
        }

        auto deleter = [sz = file_sz](char* p) {
            if (p) {
                munmap(p, sz);
            }
        };

        mmap_ptr = std::unique_ptr<char, std::function<void(char*)>>((char*)ptr, deleter);
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
    
        size_t k_offs_pos;
        size_t v_offs_pos;
        size_t k_blob_pos;
        size_t v_blob_pos;

        template <typename T>
        T read_at(size_t offset) const {
            if (offset + sizeof(T) > view.size()) {
                 return T{}; 
            }
            T val;
            std::memcpy(&val, view.data() + offset, sizeof(T));
            return val;
        }

    public:
        BlockView(std::string_view block_data) : view(block_data) {
            count = read_at<uint16_t>(0);
            uint16_t kblob_sz = read_at<uint16_t>(2);
            
            size_t offs_len = (static_cast<size_t>(count) + 1) * 2;
            
            k_offs_pos = 4;
            v_offs_pos = k_offs_pos + offs_len;
            k_blob_pos = v_offs_pos + offs_len;
            v_blob_pos = k_blob_pos + kblob_sz;
        }

        uint16_t size() const { 
            return count; 
        }

        template <typename T>
        T get_key(uint16_t idx) const {
            if (idx >= count) {
                return T{};
            }
            
            uint16_t start = read_at<uint16_t>(k_offs_pos + (idx * 2));
            uint16_t end = read_at<uint16_t>(k_offs_pos + ((idx + 1) * 2));
            
            uint16_t len = end - start;
            
            T val = T{};
            size_t copy_amount = std::min((size_t)len, sizeof(T));
            
            std::memcpy(&val, view.data() + k_blob_pos + start, copy_amount);
            
            return val;
        }

        std::string get_val_string(uint16_t idx) const {
            if (idx >= count) {
                return "";
            }
            
            uint16_t start = read_at<uint16_t>(v_offs_pos + (idx * 2));
            uint16_t end = read_at<uint16_t>(v_offs_pos + ((idx + 1) * 2));
            
            uint16_t len = end - start;
            
            return std::string(view.substr(v_blob_pos + start, len));
        }
    };

    BlockView get_block(size_t block_idx) const {
        if (block_idx * block_sz >= file_sz) {
            throw std::out_of_range("Block idx out of bounds");
        }
        
        char* block_start = mmap_ptr.get() + (block_idx * block_sz);
        return BlockView(std::string_view(block_start, block_sz));
    }
};
