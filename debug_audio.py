import sys

import librosa
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "recordings/001_국물.wav"
y, sr = librosa.load(path, sr=16000, mono=True)
print(f"len={len(y)/sr:.2f}s sr=16000 peak={np.abs(y).max():.3f} "
      f"rms={np.sqrt(np.mean(y**2)):.4f}")
print(f"DC offset(mean)={y.mean():+.5f}")

# 프레임별 RMS로 발화 구간 음량
frame = 1600  # 0.1s
rms = np.array([np.sqrt(np.mean(y[i:i+frame]**2)) for i in range(0, len(y)-frame, frame)])
print(f"frame RMS: min={rms.min():.4f} max={rms.max():.4f} median={np.median(rms):.4f}")
loud = np.where(rms > rms.max()*0.5)[0]
if len(loud):
    print(f"발화추정구간(0.1s단위): {loud[0]*0.1:.1f}s ~ {(loud[-1]+1)*0.1:.1f}s")

# 주파수 분포: 저역 hum 비중
S = np.abs(np.fft.rfft(y))
freqs = np.fft.rfftfreq(len(y), 1/sr)
def band(lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return S[m].sum()
total = S.sum() + 1e-9
print(f"energy <100Hz={band(0,100)/total:.1%}  100-300={band(100,300)/total:.1%}  "
      f"300-3400(voice)={band(300,3400)/total:.1%}  >3400={band(3400,8000)/total:.1%}")

# 클리핑 여부
clip = np.mean(np.abs(y) > 0.99)
print(f"clipping ratio={clip:.4%}")
