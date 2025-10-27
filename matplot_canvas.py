import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model_trainer import signum
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# -------- Canvas class --------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_decision_boundary(
        self, x_test_scaled, y_test, features, updated_w, updated_b
    ):
        """Recreates your plotting logic within the PySide6 canvas."""
        self.ax.clear()

        # Scatter plot
        sns.scatterplot(
            x=x_test_scaled[:, 0],
            y=x_test_scaled[:, 1],
            hue=y_test,
            palette="RdGy_r",
            legend="full",
            ax=self.ax,
        )

        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        yy, xx = np.linspace(ylim[0], ylim[1], 500), np.linspace(xlim[0], xlim[1], 500)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()])

        grid_test = np.dot(updated_w.T, xy) + updated_b
        grid_test = np.array(
            [1 if signum(output) == 1 else 0 for output in grid_test]
        ).reshape(XX.shape)

        self.ax.contourf(XX, YY, grid_test, alpha=0.1)

        self.ax.set_title("Decision Boundary on Test Set")
        self.ax.set_xlabel(features[0])
        self.ax.set_ylabel(features[1])

        self.draw()

    # def plot_signal(self, signal: 'Signal', is_discrete: bool):
    #     """Plot a single signal, either as discrete (stem) or continuous (line)."""
    #     self.ax.clear()

    #     if signal.values.size == 0:
    #         self.ax.set_title(f"{signal.name} (empty)")
    #         self.draw()
    #         return

    #     n = signal.indices()
    #     self.ax.set_title(signal.name)
    #     self.ax.set_xlabel("n")
    #     self.ax.set_ylabel("Amplitude")

    #     if is_discrete:
    #         markerline, stemlines, baseline = self.ax.stem(n, signal.values)
    #         plt.setp(markerline, color='C0')
    #         plt.setp(stemlines, color='C0')
    #     else:
    #         self.ax.plot(n, signal.values, color='C0', linewidth=1.8)

    #     self.ax.grid(True)
    #     self.draw()

    # def plot_multiple(self, signals: List['Signal'], is_discrete: bool):
    #     """Plot multiple signals in discrete or continuous mode."""
    #     self.ax.clear()

    #     if not signals:
    #         self.ax.set_title("No signals to plot")
    #         self.draw()
    #         return

    #     starts = [s.start for s in signals if s.values.size > 0]
    #     ends = [s.start + s.values.size - 1 for s in signals if s.values.size > 0]

    #     if not starts:
    #         self.ax.set_title("All signals empty")
    #         self.draw()
    #         return

    #     full_start = min(starts)
    #     full_end = max(ends)
    #     n = np.arange(full_start, full_end + 1)

    #     colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))

    #     for s, color in zip(signals, colors):
    #         if s.values.size == 0:
    #             continue

    #         arr = np.zeros(n.size)
    #         idx = s.start - full_start
    #         arr[idx:idx + s.values.size] = s.values

    #         if is_discrete:
    #             markerline, stemlines, baseline = self.ax.stem(n, arr, label=s.name)
    #             plt.setp(markerline, color=color)
    #             plt.setp(stemlines, color=color)
    #         else:
    #             self.ax.plot(n, arr, label=s.name, color=color, linewidth=1.8)

    #     self.ax.set_xlabel("n")
    #     self.ax.set_ylabel("Amplitude")
    #     self.ax.legend()
    #     self.ax.grid(True)
    #     self.draw()
