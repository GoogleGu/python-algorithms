import random
import time

import numpy as np
import cv2

from basic_sort import selection_sort, shell_sort, insertion_sort


class DataSequence:

    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 127, 255)

    MAX_IM_SIZE = 500

    def __init__(self, length,
                 time_interval=1,
                 title="Figure",
                 is_resampling=False,
                 is_sparse=False):
        self.title = title
        self.data = [x for x in range(length)]
        if is_resampling:
            self.data = random.choice(self.data, k=length)
        else:
            self.shuffle()
        if is_sparse:
            self.data = [x if random.random() < 0.3 else 0 for x in self.data]

        self.length = length

        self.set_interval(time_interval)
        self.get_figure()
        self.init_time()

        self.visualize()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def init_time(self):
        self.start = time.time()
        self.time = 0
        self.stop_timer()

    def start_timer(self):
        self.start_flag = True
        self.start = time.time()

    def stop_timer(self):
        self.start_flag = False

    def get_time(self):
        if self.start_flag:
            self.time = time.time() - self.start

    def set_interval(self, time_interval):
        self.time_interval = time_interval

    def shuffle(self):
        random.shuffle(self.data)

    def get_figure(self):
        _bar_width = 5
        figure = np.full((self.length * _bar_width, self.length * _bar_width, 3), 255, dtype=np.uint8)
        for i in range(self.length):
            val = self.data[i]
            figure[-1 - val * _bar_width:, i * _bar_width:i * _bar_width + _bar_width] = self.get_color(val, self.length)
        self._bar_width = _bar_width
        self.figure = figure
        size = _bar_width * self.length
        self.im_size = size if size < self.MAX_IM_SIZE else self.MAX_IM_SIZE

    @staticmethod
    def get_color(val, total):
        return 120 + val * 255 // (2 * total), 255 - val * 255 // (2 * total), 0

    def _set_figure(self, idx, val):
        min_col = idx * self._bar_width
        max_col = min_col + self._bar_width
        min_row = -1 - val * self._bar_width
        self.figure[:, min_col:max_col] = self.WHITE
        self.figure[min_row:, min_col:max_col] = self.get_color(val, self.length)

    def set_color(self, img, marks, color):
        for idx in marks:
            min_col = idx * self._bar_width
            max_col = min_col + self._bar_width
            min_row = -1 - self.data[idx] * self._bar_width
            img[min_row:, min_col:max_col] = color

    def mark(self, img, marks, color):
        self.set_color(img, marks, color)

    def set_val(self, idx, val):
        self.data[idx] = val
        self._set_figure(idx, val)

        self.visualize((idx,))

    def swap(self, idx1, idx2):
        self.data[idx1], self.data[idx2] = self.data[idx2], self.data[idx1]
        self._set_figure(idx1, self.data[idx1])
        self._set_figure(idx2, self.data[idx2])

        self.visualize((idx1, idx2))

    def visualize(self, mark1=None, mark2=None):
        img = self.figure.copy()
        if mark2:
            self.mark(img, mark2, self.YELLOW)
        if mark1:
            self.mark(img, mark1, self.RED)

        img = cv2.resize(img, (self.im_size, self.im_size))

        self.get_time()
        cv2.putText(img, self.title + " Time:%02.2fs" % self.time, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                    self.YELLOW, 1)

        cv2.imshow(self.title, img)

        cv2.waitKey(self.time_interval)

    def hold(self):
        for idx in range(self.length):
            self.set_val(idx, self.data[idx])

        self.set_interval(0)
        self.visualize()


if __name__ == '__main__':
    ds = DataSequence(length=50)
    ds.start_timer()
    shell_sort(ds)
    ds.stop_timer()
    ds.hold()
