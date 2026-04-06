# Minimal PyBullet demo. For RL, see line_follow_env.py and train.py.
import pybullet as p
import pybullet_data
import time

# Connect to simulator
p.connect(p.GUI)  # use p.DIRECT for no GUI

# Load default assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Add gravity
p.setGravity(0, 0, -9.8)

# Load ground plane
plane = p.loadURDF("plane.urdf")

# Load a robot (simple box or model)
robot = p.loadURDF("r2d2.urdf", [0, 0, 1])

# Run simulation
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1/240)