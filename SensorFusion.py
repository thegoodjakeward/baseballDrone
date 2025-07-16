import numpy as np
import time
from GPSReader import GPSReader
from IMUReader import IMUReader
from LocalFrame import LocalFrame
import matplotlib.pyplot as plt
import numpy as np

class GPS_IMU_KF:
    def __init__(self, dt, initial_coords=None, gps_r=5.0, init_uncertainty=100.0):
        self.dt = dt
        self.gps = GPSReader()
        self.imu = IMUReader()

        # 1) Build local ENU frame at the true origin
        lat0, lon0 = initial_coords
        self.frame = LocalFrame(lat0, lon0)

        # 2) Initialize state to that GPS fix
        x0, y0 = self.frame.latlon_to_enu(lat0, lon0)
        self.x = np.array([[x0], [y0], [0.0], [0.0]])

        # 3) Larger initial covariance → faster correction
        self.P = np.eye(4) * init_uncertainty
        
        # Process noise (tune based on IMU noise)
        q = 0.1
        self.Q = np.array([
            [dt**4/4,      0, dt**3/2,      0],
            [     0, dt**4/4,      0, dt**3/2],
            [dt**3/2,      0,    dt**2,      0],
            [     0, dt**3/2,      0,    dt**2]
        ]) * q

        # Measurement matrix (GPS measures position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Measurement noise (tune based on GPS spec)
        r_pos = 5.0  # meters
        self.R = np.eye(2) * (r_pos**2)

        # Identity matrix
        self.I = np.eye(4)

    def predict(self, ax, ay):
        dt = self.dt

        # State transition matrix
        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])

        # Control matrix
        B = np.array([
            [0.5*dt**2,          0],
            [         0, 0.5*dt**2],
            [        dt,          0],
            [         0,         dt]
        ])

        u = np.array([[ax], [ay]])

        # Predict state and covariance
        self.x = F.dot(self.x) + B.dot(u)
        self.P = F.dot(self.P).dot(F.T) + self.Q

    def update(self, meas_x, meas_y):
        z = np.array([[meas_x], [meas_y]])
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        self.P = (self.I - K.dot(self.H)).dot(self.P)

    def get_state(self):
        return self.x.flatten()

# --- Main Loop Example ---

log = {
    "t": [], "Ax_raw": [], "Ay_raw": [],
    "Ax": [], "Ay": [],
    "vx_pred": [], "vy_pred": [],
    "vx_fuse": [], "vy_fuse": [],
    "res_x": [], "res_y": [],
    "Pvx": [], "Pvy": []
}
last_time = time.time()

# Wait for a valid GPS fix
print("Waiting for initial GPS fix...")
init_coords = None
while not init_coords:
    fix = GPSReader().read_coordinates()
    if fix:
        init_coords = fix
    time.sleep(0.5)

print(f"Initial GPS fix: {init_coords}")

# Initialize local coordinate frame
frame = LocalFrame(init_coords[0], init_coords[1])

# Pass init fix to Kalman filter
kf = GPS_IMU_KF(dt=0.1, initial_coords=init_coords, gps_r=2.0, init_uncertainty=100.0)


print("Calibrating IMU...")
samples = []
for _ in range(100):
    accel = kf.imu.get_acceleration()
    if accel:
        samples.append(accel)
    time.sleep(0.01)

avg_ax = sum(a[0] for a in samples) / len(samples)
avg_ay = sum(a[1] for a in samples) / len(samples)

print(f"IMU bias: ax={avg_ax:.4f}, ay={avg_ay:.4f}")
print("Starting fusion – press Ctrl-C to stop and plot.")
try:
    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        # --- 1. Read & bias-correct acceleration ---
        ax_raw, ay_raw, _ = kf.imu.get_acceleration()
        ax = ax_raw - avg_ax
        ay = ay_raw - avg_ay

        # --- 2. Predict step ---
        kf.predict(ax, ay)
        vx_pred, vy_pred = kf.x[2,0], kf.x[3,0]

        # --- 3. GPS update (if valid) & residual ---
        coords = kf.gps.read_coordinates()
        if coords:
            x_m, y_m = kf.frame.latlon_to_enu(*coords)
            # innovation = measurement – predicted position
            residual = np.array([[x_m],[y_m]]) - kf.H.dot(kf.x)
            res_x, res_y = residual[0,0], residual[1,0]
            kf.update(x_m, y_m)
        else:
            res_x = res_y = np.nan

        vx_fuse, vy_fuse = kf.x[2,0], kf.x[3,0]

        # --- 4. Log everything ---
        log["t"].append(now)
        log["Ax_raw"].append(ax_raw)
        log["Ay_raw"].append(ay_raw)
        log["Ax"].append(ax)
        log["Ay"].append(ay)
        log["vx_pred"].append(vx_pred)
        log["vy_pred"].append(vy_pred)
        log["vx_fuse"].append(vx_fuse)
        log["vy_fuse"].append(vy_fuse)
        log["res_x"].append(res_x)
        log["res_y"].append(res_y)
        log["Pvx"].append(kf.P[2,2])
        log["Pvy"].append(kf.P[3,3])

        # 5. Print or otherwise inspect
        print(f"x={kf.x[0,0]:.2f}, y={kf.x[1,0]:.2f}, "
              f"vx={vx_fuse:.2f}, vy={vy_fuse:.2f}")

        # 6. Sleep to maintain ~dt timing
        time.sleep(max(0, kf.dt - (time.time() - now)))

except KeyboardInterrupt:
    pass  # fall through to plotting

# --- 7. Plot the logged data ---
import matplotlib.pyplot as plt
import numpy as np

t0 = log["t"][0]
t  = np.array(log["t"]) - t0

plt.figure()
plt.plot(t, log["Ax_raw"], label="Ax raw")
plt.plot(t, log["Ax"],     label="Ax corr")
plt.title("Acceleration X"); plt.xlabel("Time [s]")
plt.legend()

plt.figure()
plt.plot(t, log["vx_pred"], label="vx_pred")
plt.plot(t, log["vx_fuse"], label="vx_fuse")
plt.title("Velocity X"); plt.xlabel("Time [s]")
plt.legend()

plt.figure()
plt.plot(t, log["res_x"], '.', label="res_x")
plt.title("GPS Innovation X"); plt.xlabel("Time [s]")
plt.legend()

plt.figure()
plt.plot(t, log["Pvx"], label="P[2,2]")
plt.plot(t, log["Pvy"], label="P[3,3]")
plt.title("Velocity Covariance"); plt.xlabel("Time [s]")
plt.legend()

plt.show()