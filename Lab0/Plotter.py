import pylab as plt
import matplotlib

matplotlib.use('TkAgg')
plt.ion()


class Plotter:
    def __init__(self, title):
        self.xs = []
        self.ys = []
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.line, = self.ax.plot(self.xs, self.ys, 'r-')
        self.ax.set_title(title)
        self.ax.set_xlabel('epochs')
        self.ax.set_ylabel('accuracy')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        plt.draw()
        plt.pause(0.01)

    def add_data(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        self.line.set_xdata(self.xs)
        self.line.set_ydata(self.ys)
        self.ax.set_xlim(0, self.xs[-1])
        plt.draw()
        plt.pause(0.01)

    def save(self, name, dir_name='Plots/'):
        plt.savefig(dir_name + name)
