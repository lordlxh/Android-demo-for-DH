#pragma once
#include "jmat.h"
#include <vector>
#include <mutex>

class MBufCache {
protected:
    int m_block_width;
    int m_block_height;
    int m_block_range;
    int m_start_padding;
    std::mutex* m_lock;
    std::vector<JMat*> m_blocks;
    int m_tagarr[512];
public:
    JMat* secBuf(int sec);
    JMat* inxBuf(int inx);
    int* tagarr();
    MBufCache(int block_width, int block_height, int start_padding, int block_range);
    virtual ~MBufCache();
};

class MBnfCache: public MBufCache {
public:
    MBnfCache();
    virtual ~MBnfCache();
};
