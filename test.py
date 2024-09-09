import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os

# Project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Original Translation
mat_contents = sio.loadmat(os.path.join(BASE_DIR, "results/pose_estimation/Felipe1.mat"))
max_0 = np.max(mat_contents['simData'][0:,:3])
X0 = mat_contents['simData'][0:, 12]
Y0 = mat_contents['simData'][0:, 13]
Z0 = mat_contents['simData'][0:, 14]

# Results file with np.eye(4)
RESULT_FILE_1 = np.load(os.path.join(BASE_DIR, "results/pose_estimation/pose_Felipe1_500_1.npy"))
max_1 = np.max(RESULT_FILE_1)
fator_1 = max_0 / max_1
X1 = RESULT_FILE_1[0:, 0, 3] * fator_1
Y1 = RESULT_FILE_1[0:, 1, 3] * -fator_1
Z1 = RESULT_FILE_1[0:, 2, 3] * fator_1

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X0, Y0,  c='r', marker='o', label='Original')
ax.scatter(X1, Y1,  c='g', marker='x', label='My Results')

ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.set_zlabel('Z')

ax.legend()
plt.show()
