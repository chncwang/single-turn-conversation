#ifndef N3LDG_CUDA_MEMORY_POOL_H
#define N3LDG_CUDA_MEMORY_POOL_H

#include <vector>
#include <sstream>
#include <list>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <json/json.h>
#include <boost/format.hpp>

using namespace std;

namespace n3ldg_cuda {

struct MemoryBlock {
    void *p;
    int64_t size;
    void *buddy = nullptr;
    int id;

    MemoryBlock() {
        abort();
    }

    MemoryBlock(void *p, int size, void *buddy = nullptr) {
        static int global_id;
        if (size <= 0 || size & (size - 1) != 0) {
            std::cerr << "illegal size:" << size << std::endl;
        }
        this->p = p;
        this->size = size;
        this->buddy = buddy;
        this->id = global_id++;
    }

    string toString() const {
        Json::Value json;
        stringstream p_stream;
        p_stream << p;
        json["p"] = p_stream.str();
        json["size"] = size;
        stringstream buddy_stream;
        buddy_stream << buddy;
        json["buddy"] = buddy_stream.str();
        json["id"] = id;
        return Json::writeString(Json::StreamWriterBuilder(), json);
    }
};

class MemoryPool {
public:
    MemoryPool(const MemoryPool &) = delete;
    static MemoryPool& Ins();

    cudaError_t Malloc(void **p, int size);
    cudaError_t Free(void *p);

    string toString() const {
        Json::Value free_blocks_json;
        int i = 0;
        for (auto & v : free_blocks_) {
            Json::Value json;
            int j = 0;
            for (auto &block : v) {
                json[j++] = block.toString();
            }
            if (!v.empty()) {
                Json::Value json_and_index;
                json_and_index["i"] = i;
                json_and_index["v"] = json;
                free_blocks_json.append(json_and_index);
            }
            i++;
        }
        return Json::writeString(Json::StreamWriterBuilder(), free_blocks_json);
    }

private:
    MemoryPool() = default;
    std::vector<std::vector<MemoryBlock>> free_blocks_;
    std::unordered_map<void *, MemoryBlock> busy_blocks_;
};

}

#endif
