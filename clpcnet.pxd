cdef extern from "src/lpcnet.h" nogil:
    ctypedef struct LPCNetState:
        pass

    ctypedef struct LPCNetDecState:
        pass

    ctypedef struct LPCNetEncState:
        pass
    
    cdef const int NB_TOTAL_FEATURES
    cdef const int NB_FEATURES
    cdef const int LPCNET_FRAME_SIZE

    LPCNetState * lpcnet_create()
    LPCNetEncState * lpcnet_encoder_create()

    void lpcnet_synthesize(LPCNetState *lpcnet, const float *feature, 
                            short *output, int N)

    int lpcnet_compute_features(LPCNetEncState *st, const short *pcm,
                                    float features[4][NB_TOTAL_FEATURES])

    int lpcnet_init(LPCNetState *lpcnet)
    int lpcnet_encoder_init(LPCNetEncState *st)


