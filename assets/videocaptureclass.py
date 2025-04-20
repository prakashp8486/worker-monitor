import sys
sys.dont_write_bytecode = True

import threading
import queue
import cv2
import time

class VideoCapture:
    def __init__(self, camera_link):
        self.camera_link = camera_link
        self.cap = cv2.VideoCapture()
        self.cap.open(camera_link)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            try:
                ret, frame = self.cap.read()
                if ret:
                    if not self.q.empty():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                    self.q.put(frame)
                else:
                    print("Frame capture error. Reopening camera.")
                    self.cap.open(self.camera_link)
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error reading frames in VideoCapture class: {e}")

    def read(self):
        try:
            return self.q.get()
        except Exception as e:
            print(f"Error getting frame from queue: {e}")
            return None