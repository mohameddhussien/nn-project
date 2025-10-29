import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# -------- Canvas class --------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        super().__init__(Figure(figsize=(6, 4)))
        self.setParent(parent)
        self.ax = self.figure.add_subplot(1, 1, 1)

    def _reset_figure(self, figsize=(6, 4)):
        # self.figure.clear()
        # self.figure.set_size_inches(*figsize)
        self.ax.clear()
        self.ax = self.figure.add_subplot(1, 1, 1)

    def plot_decision_boundary(self, x_test_scaled, y_test, features, w, b):
        self._reset_figure(figsize=(6, 4))

        sns.scatterplot(
            x=x_test_scaled[:, 0],
            y=x_test_scaled[:, 1],
            hue=y_test,
            palette="RdGy_r",
            legend="full",
            ax=self.ax,
        )

        xlim = self.ax.get_xlim()
        x_vals = np.linspace(xlim[0], xlim[1])
        y_vals = -(w[0] * x_vals + b) / w[1]

        self.ax.plot(x_vals, y_vals, "k-", linewidth=2)
        self.ax.set_title("Linear Decision Boundary")
        self.ax.set_xlabel(features[0])
        self.ax.set_ylabel(features[1])

        self.draw()

    def plot_confusion_matrix(self, y_pred_test, y_test):
        # Confusion Matrix "btlooo fazlkaaaa b2a wsebona f7alna"
        self._reset_figure(figsize=(4, 4))

        TP = np.sum((y_pred_test == 1) & (y_test == 1))
        TN = np.sum((y_pred_test == 0) & (y_test == 0))
        FP = np.sum((y_pred_test == 1) & (y_test == 0))
        FN = np.sum((y_pred_test == 0) & (y_test == 1))

        conf_matrix = np.array([[TN, FP], [FN, TP]])

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        im = self.ax.imshow(conf_matrix, cmap='Blues')

        for i in range(2):
            for j in range(2):
                self.ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black", fontsize=12, fontweight="bold")

        self.ax.set_xticks([0, 1])
        self.ax.set_yticks([0, 1])
        self.ax.set_xticklabels(['Predicted 1', 'Predicted 0'])
        self.ax.set_yticklabels(['Actual 1', 'Actual 0'])
        self.ax.set_xlabel('Predicted Label', fontsize=12)
        self.ax.set_ylabel('True Label', fontsize=12)
        self.ax.set_title('Confusion Matrix (TP-FN / FP-TN Format)', fontsize=14, fontweight="bold")

        self.figure.colorbar(im, ax=self.ax, pad=0.1)
        self.figure.tight_layout()
        self.draw()

        return conf_matrix, precision, recall, f1
