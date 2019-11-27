# LPCNet Vocoder implemented by Cython

# Install
```python
python setup.py install
```

# Usage
* extract 20-dim feature from wav
```python
import lpcnet
import soundfile as sf
import numpy as np
s, sr = sf.read('test.wav')
s = s / np.max(np.abs(s)) * 32768 * 0.95
s = s.astype(np.int16)
extractor = lpcnet.FeatureExtractor()
feature = extractor.compute_feature(s)
```

* synthesize wav from 20-dim feature
```python
import lpcnet
synthesizer = lpcnet.Synthesizer()
feature = np.load('feature.npy')
feature = feature.astype(np.float32)
pcm = synthesizer.synthesis(feat) 
```


# Train a new model
* See [LPCNet: https://github.com/IMLHF/LPCNet](https://github.com/IMLHF/LPCNet)

# 
Author [@jmvalin](https://github.com/jmvalin) [@vBaiCai](https://github.com/vBaiCai)