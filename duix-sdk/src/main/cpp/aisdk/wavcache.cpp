#include "aicommon.h"
#include "wavcache.h"

JMat* MBufCache::secBuf(int sec) {
    sec += m_start_padding;
    JMat* result = NULL;
    m_lock->lock();
    if (sec < m_blocks.size()) {
        result = m_blocks[sec];
    }
    else {
        result = new JMat(m_block_width, m_block_height, 1);
        m_blocks.push_back(result);
    }
    m_lock->unlock();
    return result;
}

JMat* MBufCache::inxBuf(int index) {
    JMat* result = new JMat(m_block_width, m_block_height * m_block_range, 1);
    float* dst = result->fdata();
    const int block_size = m_block_width * m_block_height;
    const size_t block_memory_size = block_size * sizeof(float);
    const int end_index = std::min<int>(index + m_block_range, m_blocks.size());
    for (int i = index; i < end_index; i++) {
        memcpy(dst, m_blocks[i]->fdata(), block_memory_size);
        dst += block_size;
    }
    return result;
}

int* MBufCache::tagarr() {
    return m_tagarr;
}

MBufCache::MBufCache(int block_width, int block_height, int start_padding, int block_range) {
    m_lock = new std::mutex();
    m_block_width = block_width;
    m_block_height = block_height;
    m_block_range = block_range;
    m_start_padding = start_padding;
    const size_t block_memory_size = block_width * block_height;
    for (int i = 0; i < start_padding; i++) {
        JMat* block = new JMat(m_block_width, m_block_height, 1);
        memset(block->fdata(), 0, block_memory_size * sizeof(float));
        m_blocks.push_back(block);
    }
    memset(m_tagarr, 0, 512 * sizeof(int));
}

MBufCache::~MBufCache() {
    m_lock->lock();
    for (auto block : m_blocks) {
        delete block;
    }
    m_blocks.clear();
    m_lock->unlock();
    delete m_lock;
}

MBnfCache::MBnfCache() : MBufCache(MFCC_BNFCHUNK, MFCC_BNFBASE, 8, 16) {
}

MBnfCache::~MBnfCache() {
}

#ifdef _WAVTEST_
int main(int argc, char** argv) {
    MBnfCache cache;
    for (int k = 0; k < 100; k++) {
        JMat* mat = cache.secBuf(k);
    }
    for (int k = 816; k < 817; k++) {
        printf("#%d# \n", k);
        JMat* mat = cache.inxBuf(k);
        JMat cm = mat->clone();
        delete mat;
    }
    return 0;
}
#endif
