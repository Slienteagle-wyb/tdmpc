import os
os.environ['MUJOCO_GL'] = 'egl'
import time
from dm_control import suite
env = suite.load('cartpole', 'swingup')

st = time.time()
for i in range(100):
    pixels = env.physics.render()
time_used = time.time() - st
print(time_used)