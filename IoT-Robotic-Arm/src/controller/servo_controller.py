# S1A56 S3A100 S5A116 S7A90 S9A46 - BOTTOM LEFT
# S1A19 S3A57 S5A55 S7A90 S9A39 - TOP LEFT
# S1A25 S3A71 S5A67 S7A90 S9A29 - TOP MIDDLE
# S1A23 S3A51 S5A71 S7A90 S9A46 - TOP RIGHT

import time

import serial

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ServoControl:
    def __init__(self, COM: str = "COM5"):
        self.basePosition = 90

        self.positionData = {
            "17": "S2A58S3A150S4A115S5A105",
            "18": "S2A52S3A145S4A115S5A105",
            "19": "S2A50S3A138S4A115S5A105",
            "20": "S2A50S3A138S4A115S5A105",
            "21": "S2A42S3A128S4A120S5A105",
            "22": "S2A40S3A124S4A120S5A105",
            "23": "S2A38S3A122S4A120S5A105",
            "24": "S2A26S3A103S4A120S5A105",
            "25": "S2A25S3A100S4A115S5A105",
            "26": "S2A10S3A88S4A115S5A105`",
            "rest": "S1A75S2A80S3A145S4A70S5A90",
            "neutral": "S1A90S2A90S3A90S4A90",
            "paperDisposal": "S1A70S2A120S3A40",
            "plasticDisposal": "S1A50S2A120S3A40",
            "metalDisposal": "S1A100S2A120S3A40",
        }
        self.ser = serial.Serial(port=COM, baudrate=9600, timeout=1)
        (f"Connected to Arduino on {COM}")
        self.ser.write("S1A90S3A90S5A90S7A90S9A90\n".encode("utf-8"))
        logger.info("Servos set to initial postions (90 degrees)")

    def read_serial(self):
        """
        Continuously reads data from the serial monitor and prints it to the console.
        Stops and returns False if the keyword "OK" appears in the incoming data.

        :return: False when "OK" is detected else True
        """

        # return False # For testing purposes, This will take allow to read data from camera continously

        try:
            while True:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode("utf-8").strip()
                    logger.debug(line)

                    if "OK" in line:
                        return False
                    else:
                        return True
        except KeyboardInterrupt:
            logger.error("\nSerial reading stopped manually.")
            return
        except Exception as e:
            logger.error(f"Error: {e}")
            return

    def moveServoSingle(self, servo: int, angle: int):
        """FOR SENDING COMMANDS TO ONE SERVO"""
        full_command = f"S{servo}A{angle}\n"
        self.ser.write(full_command.encode("utf-8"))
        time.sleep(0.1)
        while self.read_serial():
            time.sleep(0.1)  # Wait for the servo to finish moving
        logger.info(f"Servo {servo} moved to {angle} degrees")

    def moveServoBatch(self, servo_angle_list: str):
        """FOR SENDING COMMANDS TO MULTIPLE SERVOS"""
        servo_angle_list += "\n"
        self.ser.write(servo_angle_list.encode("utf-8"))
        time.sleep(0.1)
        while self.read_serial():
            time.sleep(0.1)  # Wait for the servo to finish moving
        logger.info("Servo moved sucessfully")

    def closeGripper(self):
        self.moveServoSingle(5, 40)

    def openGripper(self):
        self.moveServoSingle(5, 100)

    def setCircle(self, circle: str):
        self.moveServoBatch(self.positionData[circle])

    def setNeutralPosition(self):
        """Set servos to neutral position"""
        self.moveServoBatch(self.positionData["neutral"])

    def disposeDegradableWaste(self):
        """Move to position for degradable waste disposal"""
        self.moveServoBatch(self.positionData["degradableDisposal"])

    def disposeNonDegradableWaste(self):
        """Move to position for non-degradable waste disposal"""
        self.moveServoBatch(self.positionData["nonDegradableDisposal"])

    def setRestPostion(self):
        """Set servos to rest position"""
        self.moveServoBatch(self.positionData["rest"])

    def baseServoLeft(self):
        self.basePosition += 1
        self.moveServoSingle(1, self.basePosition)

    def baseServoRight(self):
        self.basePosition -= 1
        self.moveServoSingle(1, self.basePosition)

    def close(self):
        self.ser.close()
