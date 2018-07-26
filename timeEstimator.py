from __future__ import division
import time
import datetime


class TimeEstimator:
    def __init__(self, multiplier):
        self.times = []
        self.currentTime = None
        self.multiplier = float(multiplier)

    def get(self):
        if len(self.times) < 1:
            return 0
        return sum(self.times) / len(self.times) * self.multiplier

    def start(self):
        self.currentTime = time.time()

    def capture(self):
        if self.currentTime is None:
            print('You must call TimeEstimator.start before calling capture (or after each capture)')
            return
        self.times.append(time.time() - self.currentTime)
        # self.currentTime = None

        self.currentTime = time.time()

    def __str__(self):
        return str(datetime.timedelta(seconds=self.get()))

    def reset(self):
        self.currentTime = time.time()


if __name__ == '__main__':
    t = TimeEstimator(5)

    t.start()
    time.sleep(1)
    t.capture()
    t.start()
    time.sleep(3)
    t.capture()

    print(t)
