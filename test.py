import lpcnet
import numpy as np
import array

N_LPC_FEATURES = 55


def expand_feature(compact_feature):
    n_frames = compact_feature.shape[0]
    feature = np.zeros([n_frames, N_LPC_FEATURES], dtype='float32')
    feature[:, :18] = compact_feature[:, :18]
    feature[:, 36:38] = compact_feature[:, 18:]
    return feature


def compress_feature(feature):
    bfcc = feature[:, :18]
    pitch = feature[:, 36:38]
    compact_feature = np.concatnate([bfcc, pitch], axis=-1)
    return compact_feature


synthesizer = lpcnet.Synthesizer()
feature = np.load('sample.npy')
feature = expand_feature(feature)
feature = feature.astype(np.float32)

import time
tic = time.time()
pcm = synthesizer.synthesis(feature)
pcm.tofile('out.pcm')
print('[synthesize]: ', time.time() - tic)


pcm = np.fromfile('1.pcm', dtype=np.int16)
pcm.tofile('1-save.pcm')
extractor = lpcnet.FeatureExtractor()
tic = time.time()
feature = extractor.compute_feature(pcm)
print('[compute feature]: ', time.time() - tic)

tic = time.time()
pcm = synthesizer.synthesis(feature)
pcm.tofile('resyn.pcm')
print('[resynthesis]: ', time.time() - tic)



