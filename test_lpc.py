import lpcnet
import soundfile as sf
import numpy as np
s, sr = sf.read('000001.wav')
s = s / np.max(np.abs(s)) * 32768 * 0.95
s = s.astype(np.int16)
print(1)
extractor = lpcnet.FeatureExtractor()
print(2,s)
feature = extractor.compute_feature(s)
print(3)
print(feature)
