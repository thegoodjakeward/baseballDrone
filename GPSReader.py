import serial
import pynmea2

class GPSReader:
    def __init__(self, port="/dev/serial0", baudrate=9600, timeout=0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        try:
            self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        except serial.SerialException as e:
            print(f"[GPSReader] Serial init error: {e}")
            self.ser = None

    def read_coordinates(self):
        if not self.ser or not self.ser.is_open:
            print("[GPSReader] Serial port not open")
            return None

        try:
            line = self.ser.readline().decode('ascii', errors='replace')
            if line.startswith('$GPRMC'):
                msg = pynmea2.parse(line)
                if msg.status != 'A':
                    # GPS fix is invalid or not available
                    return None
                return (msg.latitude, msg.longitude)
        except pynmea2.ParseError:
            return None
        except Exception as e:
            print(f"[GPSReader] Read error: {e}")
            return None


    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

# --- Example Usage ---
if __name__ == "__main__":
    gps = GPSReader()
    try:
        while True:
            coords = gps.read_coordinates()
            if coords:
                lat, lon = coords
                print(f"Latitude: {lat}, Longitude: {lon}")
    except KeyboardInterrupt:
        pass
    finally:
        gps.close()
