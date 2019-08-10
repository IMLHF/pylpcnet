# distutils: sources = src/lpcnet.c  src/nnet.c src/nnet_data.c src/lpcnet_dec.c src/ceps_codebooks.c src/common.c src/lpcnet_enc.c src/celt_lpc.c src/pitch.c src/freq.c src/kiss_fft.c
# distutils: include_dirs = src/

cimport clpcnet
from libc.string cimport memcpy, memset
from cpython cimport array
import numpy as np


def expand_feature(compact_feature):
    n_frames = compact_feature.shape[0]
    feature = np.zeros([n_frames, clpcnet.NB_TOTAL_FEATURES], dtype='float32')
    feature[:, :18] = compact_feature[:, :18]
    feature[:, 36:38] = compact_feature[:, 18:]
    return feature


def compress_feature(feature):
    bfcc = feature[:, :18]
    pitch = feature[:, 36:38]
    compact_feature = np.concatenate([bfcc, pitch], axis=-1)
    return compact_feature


cdef class Synthesizer:
    cdef clpcnet.LPCNetState *_c_net

    def __cinit__(self):
        self._c_net = clpcnet.lpcnet_create();

    def synthesis(self, feature, compact=True):
        if compact:
            feature = expand_feature(feature)

        n_frames = feature.shape[0]
        if not feature.flags['C_CONTIGUOUS']:
            feature = np.ascontiguousarray(feature)
        pcm = np.zeros([n_frames, clpcnet.LPCNET_FRAME_SIZE], dtype=np.int16)

        cdef float[:, :] feature_mview = feature
        cdef short[:, :] pcm_mview = pcm
        cdef float* feature_ptr
        cdef short* pcm_ptr

        for i in range(n_frames):
            feature_ptr = &feature_mview[i][0]
            pcm_ptr = &pcm_mview[i][0]

            clpcnet.lpcnet_synthesize(self._c_net, feature_ptr,
                                        pcm_ptr, clpcnet.LPCNET_FRAME_SIZE)

        self.reset()
        return pcm.reshape(-1)

    def synthesize(self, feature_seq):
        return self.synthesis(self, feature_seq)

    def reset(self):
        clpcnet.lpcnet_init(self._c_net)


cdef class FeatureExtractor:
    cdef clpcnet.LPCNetEncState *_c_net;

    def __cinit__(self):
        self._c_net = clpcnet.lpcnet_encoder_create()

    def reset(self):
        clpcnet.lpcnet_encoder_init(self._c_net)

    def compute_feature(self, pcm, compact=True):
        if pcm.dtype != np.int16:
            raise TypeError("unsupported type {}, expected type np.int16".format(pcm.dtype))

        frame_size = clpcnet.LPCNET_FRAME_SIZE
        padding_size = (pcm.shape[0] % (frame_size * 4))
        padding_size = 0 if padding_size == 0 else ((frame_size * 4) - padding_size)
        pcm = np.concatenate([pcm, np.zeros([padding_size], dtype=np.int16)])
        n_frames = pcm.shape[0] // frame_size
        n_packets = n_frames // 4
        pcm = pcm.reshape(n_packets, frame_size * 4)
        if not pcm.flags['C_CONTIGUOUS']:
            pcm = np.ascontiguousarray(pcm)

        feature = np.zeros([n_packets, 4, clpcnet.NB_TOTAL_FEATURES], np.float32)

        cdef short[:, :] pcm_mview = pcm 
        cdef float[:, :, :] feature_mview = feature
        cdef short * pcm_ptr
        cdef float (*feature_ptr)[clpcnet.NB_TOTAL_FEATURES]

        for i in range(n_frames//4):
            pcm_ptr = &pcm_mview[i][0]
            feature_ptr = <float (*)[clpcnet.NB_TOTAL_FEATURES]>&feature_mview[i][0][0]
            clpcnet.lpcnet_compute_features(self._c_net, pcm_ptr, feature_ptr)

        feature = feature.reshape(-1, clpcnet.NB_TOTAL_FEATURES)
        if compact:
            feature = compress_feature(feature)

        self.reset()
        return feature