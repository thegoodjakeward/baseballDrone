import smbus2
import time

class IMUReader:
    def __init__(self, address=0x68, bus_id=1):
        self.address = address
        try:
            self.bus = smbus2.SMBus(bus_id)
            # Wake up MPU6050
            self.bus.write_byte_data(self.address, 0x6B, 0)
        except Exception as e:
            print(f"[IMUReader] Initialization error: {e}")
            self.bus = None

    def read_raw_data(self, reg_addr):
        if not self.bus:
            return 0
        try:
            high = self.bus.read_byte_data(self.address, reg_addr)
            low = self.bus.read_byte_data(self.address, reg_addr + 1)
            value = (high << 8) | low
            if value > 32768:
                value -= 65536
            return value
        except Exception as e:
            print(f"[IMUReader] Read error: {e}")
            return 0

    def get_acceleration(self):
        if not self.bus:
            return None
        try:
            acc_x = self.read_raw_data(0x3B)
            acc_y = self.read_raw_data(0x3D)
            acc_z = self.read_raw_data(0x3F)

            Ax = acc_x / 16384.0
            Ay = acc_y / 16384.0
            Az = acc_z / 16384.0
            return (Ax, Ay, Az)
        except Exception as e:
            print(f"[IMUReader] Accel error: {e}")
            return None

    def close(self):
        if self.bus:
            self.bus.close()

# --- Example Usage ---
if __name__ == "__main__":
    imu = IMUReader()
    try:
        while True:
            accel = imu.get_acceleration()
            if accel:
                Ax, Ay, Az = accel
                print(f"Acc X={Ax:.2f}, Y={Ay:.2f}, Z={Az:.2f}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        imu.close()
