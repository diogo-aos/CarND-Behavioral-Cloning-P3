import numpy as np
import get_data
import matplotlib.pyplot as plt

samples = get_data.get_samples()
angles = [float(s[1][3]) for s in samples]
angles = np.array(angles)

plt.hist(angles, bins=20)
plt.savefig('angle_hist.eps')
    
