#include "wenet.h"
#include <stdio.h>
#include <memory>
#include <vector>
#include "wavreader.h"
#include "face_utils.h"
#include "mfcc/mfcc.hpp"
#include "jlog.h"
#include "aicommon.h"

void Wenet::initModel(const char* modelfn){
    m_model = new OnnxModel();
    string modelpath(modelfn);
    m_model->initModel(modelpath);
    //m_model->pushName("speech",1);
    //m_model->pushName("speech_lengths",1);
    //m_model->pushName("encoder_out",0);
}

int Wenet::calcbnf(JMat* feat_mat, int n_feat, MBnfCache* bnf_cache) {
    // ======== dump fbank ========
/*    std::ofstream fbank_csv("/data/data/ai.guiji.duix.test/fbank.csv");
    fbank_csv << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < n_feat; i++) {
        for (size_t j = 0; j < 80; j++) {
            if (j > 0) {
                fbank_csv << ",";
            }
            fbank_csv << feat_mat->fdata()[i*80+j];
        }
        fbank_csv << std::endl;
    }*/
    // ============================

    auto config = m_model->config();
    // config.dump();

    config.shape_inputs[0][0] = 1;
    config.shape_inputs[0][1] = 1;
    config.shape_inputs[0][2] = WENET_WINDOW_SIZE;
    config.shape_inputs[0][3] = 80;
    config.size_inputs[0] = (
        config.shape_inputs[0][0]
        * config.shape_inputs[0][1]
        * config.shape_inputs[0][2]
        * config.shape_inputs[0][3]
    );
    config.shape_inputs[1][0] = 1;
    config.size_inputs[1] = 1;
    config.shape_inputs[2][0] = 3;
    config.shape_inputs[2][1] = 8;
    config.shape_inputs[2][2] = 16;
    config.shape_inputs[2][3] = 128;
    config.size_inputs[2] = (
        config.shape_inputs[2][0]
        * config.shape_inputs[2][1]
        * config.shape_inputs[2][2]
        * config.shape_inputs[2][3]
    );
    config.shape_inputs[3][0] = 3;
    config.shape_inputs[3][1] = 1;
    config.shape_inputs[3][2] = 512;
    config.shape_inputs[3][3] = 14;
    config.size_inputs[3] = (
        config.shape_inputs[3][0]
        * config.shape_inputs[3][1]
        * config.shape_inputs[3][2]
        * config.shape_inputs[3][3]
    );

    config.shape_outputs[0][0] = 1;
    config.shape_outputs[0][1] = MFCC_BNFBASE;
    config.shape_outputs[0][2] = MFCC_BNFCHUNK;
    config.size_outputs[0] = (
        config.shape_outputs[0][0]
        * config.shape_outputs[0][1]
        * config.shape_outputs[0][2]
    );

    config.shape_outputs[1][0] = 3;
    config.shape_outputs[1][1] = 8;
    config.shape_outputs[1][2] = 16;
    config.shape_outputs[1][3] = 128;
    config.size_outputs[1] = (
            config.shape_outputs[1][0]
            * config.shape_outputs[1][1]
            * config.shape_outputs[1][2]
            * config.shape_outputs[1][3]
    );
    config.shape_outputs[2][0] = 3;
    config.shape_outputs[2][1] = 1;
    config.shape_outputs[2][2] = 512;
    config.shape_outputs[2][3] = 14;
    config.size_outputs[2] = (
            config.shape_outputs[2][0]
            * config.shape_outputs[2][1]
            * config.shape_outputs[2][2]
            * config.shape_outputs[2][3]
    );


    config.dump();

    float* chunk = (float*)malloc(config.size_inputs[0] * sizeof(float));
    int64_t offset = 100;
    float* att_cache = (float*)malloc(config.size_inputs[2] * sizeof(float));
    float* cnn_cache = (float*)malloc(config.size_inputs[3] * sizeof(float));
    float* r_att_cache = (float*)malloc(config.size_outputs[1] * sizeof(float));
    float* r_cnn_cache = (float*)malloc(config.size_outputs[2] * sizeof(float));
    memset(att_cache, 0, config.size_inputs[2] * sizeof(float));
    memset(cnn_cache, 0, config.size_inputs[3] * sizeof(float));

    // -------- test data --------
   // float chunk[1][1][67][80];
    //int64_t offset[1] = {100};
/*    float att_cache[3][8][16][128];
    float cnn_cache[3][1][512][14];
    float r_att_cache[3][8][16][128];
    float r_cnn_cache[3][1][512][14];*/
    //memset(chunk, 0, 1 * 1 * 67 * 80 * sizeof(float));
    // ---------------------------


    void* array_in[] = { chunk, &offset, att_cache, cnn_cache};
    void* array_out[] = { nullptr,r_att_cache,r_cnn_cache};
    const char* names_in[] = { "chunk", "offset", "att_cache", "cnn_cache",NULL};
    const char* names_out[] = { "output","r_att_cache","r_cnn_cache",NULL};
    config.names_in = names_in;
    config.names_out = names_out;
    //for (char** p = const_cast<char **>(names_in); *p != NULL; p++) config.name_inputs.push_back(*p);
    //for (char** p = const_cast<char **>(names_out); *p != NULL; p++) config.name_outputs.push_back(*p);

    for (int i = 0; i <= n_feat - WENET_WINDOW_SIZE; i += WENET_STRIDE) {
        const float* feature_start = feat_mat->fdata() + i*80;
        const int feature_length = std::min<int>(n_feat - i, WENET_WINDOW_SIZE);
        memset(chunk, 0, config.size_inputs[0] * sizeof(float));
        memcpy(chunk, feature_start, feature_length * 80* sizeof(float));
        array_out[0] = bnf_cache->secBuf(i / WENET_STRIDE)->fdata();
        int result = m_model->runModel(array_in, array_out, NULL, &config);
        if (result != 0) {
            return result;
        }
    }

    free(chunk);
    free(att_cache);
    free(cnn_cache);
    free(r_att_cache);
    free(r_cnn_cache);

    // ======== dump wenet ========
/*    std::ofstream wenet_csv("/data/data/ai.guiji.duix.test/wenet.csv");
    wenet_csv << std::fixed << std::setprecision(6);
    for (int k = 0; k <= n_feat - WENET_WINDOW_SIZE; k += WENET_STRIDE) {
        const auto mat = bnf_cache->secBuf(k / WENET_STRIDE);
        for (size_t i = 0; i < MFCC_BNFBASE; i++) {
            for (size_t j = 0; j < MFCC_BNFCHUNK; j++) {
                if (j > 0) {
                    wenet_csv << ",";
                }
                wenet_csv << *mat->fitem(i, j);
            }
            wenet_csv << std::endl;
        }
    }*/
    // ============================

    return 0;
}

Wenet::Wenet(const char* modeldir,const char* modelid){
    char path[1024];
    sprintf(path,"%s/%s.onnx",modeldir,modelid);
    initModel((const char*)path);
}

Wenet::Wenet(const char* modelfn){
    initModel(modelfn);
}

Wenet::~Wenet(){
    delete m_model;
}

//int Wenet::nextwav(const char* wavfile,JMat** pmat){
int Wenet::nextwav(const char* wavfile,MBnfCache* bnfcache){
/*
    int     m_pcmsample = 0;
    JBuf   *m_pcmbuf = nullptr;
    JMat   *m_wavmat = nullptr;
    JMat   *m_melmat = nullptr;
    //JMat   *m_bnfmat = nullptr;

    int format, channels, sr, bits_per_sample;
    unsigned int data_length;
    void* fhnd = wav_read_open(wavfile);
    if(!fhnd)return -1;
    int res = wav_get_header(fhnd, &format, &channels, &sr, &bits_per_sample, &data_length);
    if(data_length<1) {
        wav_read_close(fhnd);
        return -2;
    }
    LOGE("data len %d\n",data_length);
    m_pcmbuf = new JBuf(data_length);
    int rst = wav_read_data(fhnd,(unsigned char*)m_pcmbuf->data(),data_length);
    wav_read_close(fhnd);
    int wavsample = data_length/2;
    m_pcmsample = wavsample + 2*MFCC_OFFSET;
    int seca = m_pcmsample / MFCC_WAVCHUNK;
    int secb = m_pcmsample % MFCC_WAVCHUNK;
    if(secb>0){
        //m_pcmsample = wavsample + 2*MFCC_OFFSET + MFCC_WAVCHUNK - secb;
        //seca++;
    }
    int mellast = secb?(secb /160 +1):0;
    int bnflast = secb?((mellast*0.25f)-0.75f):0;

    int wavsize = seca*MFCC_WAVCHUNK + secb;
    int melsize = seca*MFCC_MELBASE+mellast;
    int bnfsize = seca*MFCC_BNFBASE+bnflast;

    int calcsize = seca+1;

    m_wavmat = new JMat(MFCC_WAVCHUNK,calcsize,1);
    m_wavmat->zeros();
    short* ps = (short*)m_pcmbuf->data();
    float* pd = (float*)m_wavmat->data();
    float* pf = pd+MFCC_OFFSET;
    for(int k=0;k<wavsample;k++){
        *pf++ = (float)(*ps++/ 32767.f);
    }
    m_melmat = new JMat(MFCC_MELCHUNK,MFCC_MELBASE*calcsize,1);
    m_melmat->zeros();
    //m_bnfmat = new JMat(MFCC_BNFCHUNK,MFCC_BNFBASE*calcsize,1);
    //m_bnfmat->zeros();
    //
    //printf("===seca %d secb %d mellast %d\n",seca,secb,mellast);
    //m_bnfmat = new JMat(
    calcmfcc(m_wavmat,m_melmat);
    float* mel = m_melmat->fdata();
    for(int k=0;k<seca;k++){
        float* bnf = bnfcache->secBuf(k)->fdata();
        calcbnf(mel,MFCC_MELBASE,bnf,MFCC_BNFBASE);
        //dumpfloat(bnf,10);
        mel+=MFCC_MELBASE*MFCC_MELCHUNK;
        //bnf+=MFCC_BNFBASE*MFCC_BNFCHUNK;
    }
    if(mellast){
        //fix last
        int inxsec = seca ;//seca?(seca+1):0;
        printf("===indexsec %d\n",inxsec);
        float* bnf = bnfcache->secBuf(inxsec)->fdata();
        calcbnf(mel,mellast,bnf,bnflast);
        //dumpfloat(bnf,10);
        //calcbnf(mel,MFCC_MELBASE,bnf,MFCC_BNFBASE);
    }
    int* arr = bnfcache->tagarr();
    //
    arr[0] = wavsize;
    arr[1] = m_pcmsample;
    arr[2] = seca;
    arr[3] = secb;
    float secs = wavsample *1.0f/ MFCC_RATE;
    int bnfblock = secs*MFCC_FPS;
    if(bnfblock>(bnfsize-10))bnfblock = bnfsize-10;
    arr[4] = melsize;
    arr[5] = bnfsize;
    arr[6] = bnfblock;
    /*
    for(int k=0;k<10;k++){
        float* bnf = m_bnfmat->frow(k);
        printf("==%d =bnf %f\n",k,*bnf);
    }

    pmat = m_bnfmat;
    delete m_pcmbuf;
    delete m_wavmat;
    delete m_melmat ;
    return bnfblock;
    */
    return 0;
}

float* Wenet::nextbnf(JMat* bnfmat,int index){
    int* arr = bnfmat->tagarr();
    int bnfsize = arr[5] ;
    int bnfblock = arr[6] ;
    LOGD("===index %d bnfsize %d bnfblock %d\n",index,bnfsize,bnfblock);
    if(bnfblock>bnfsize)return NULL;
    if(index>=bnfblock)return NULL;
    float* buf = bnfmat->fdata();
    buf += index*MFCC_BNFCHUNK+MFCC_BNFCHUNK;
    return buf;
}

int Wenet::calcmfcc(float* fwav,float* mel2){
    int rst = 0;
    int melcnt = MFCC_WAVCHUNK/160+1;
    rst = log_mel(fwav,MFCC_WAVCHUNK, 16000,mel2);
    return rst;
}

int Wenet::calcmfcc(JMat* mwav,JMat* mmel){
    int rst = 0;
    int melcnt = MFCC_WAVCHUNK/160+1;
    for(size_t k=0;k<mwav->height();k++){
        float* fwav = mwav->frow(k);
        float* mel2 = mmel->frow(k);
        rst = log_mel(fwav,MFCC_WAVCHUNK, 16000,mel2);
    }
    return rst;
}

#ifdef WENET_MAIN

int main(int argc,char** argv){
    Wenet net("../model","wenet");
    net->nextwav("../mybin/a.wav");
    return 0;
}
#endif
