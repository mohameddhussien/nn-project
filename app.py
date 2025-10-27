from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QLineEdit, QComboBox, QRadioButton, QPushButton, QGroupBox,
    QButtonGroup, QMessageBox, QFileDialog, QTextEdit
)
import sys
import os
import numpy as np
from matplot_canvas import MplCanvas
from model_trainer import (
    train, initialize_weights_and_bias, preprocess_data, signum
)


class NNTrainerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Training - Perceptron / Adaline")
        self.resize(1000, 700)

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)

        # --- File Loader ---
        file_loader_layout = QHBoxLayout()
        self.file_path_label = QLabel("No dataset loaded.")
        self.load_file_button = QPushButton("Load Dataset")
        self.load_file_button.clicked.connect(self._load_dataset)
        file_loader_layout.addWidget(self.file_path_label)
        file_loader_layout.addWidget(self.load_file_button)
        main_layout.addLayout(file_loader_layout)

        # --- Tab Group ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Add Tabs
        self._add_model_tuning_tab()
        self._add_prediction_tab()
        self._add_evaluation_tab()

    # ==================================================
    # TAB 1: Model Tuning
    # ==================================================
    def _add_model_tuning_tab(self):
        tuning_tab = QWidget()
        layout = QVBoxLayout(tuning_tab)

        # --- Feature Selection (comma-separated) ---
        feat_layout = QHBoxLayout()
        feat_layout.addWidget(QLabel("Select 2 features (comma separated):"))
        self.feature_input = QLineEdit()
        self.feature_input.setPlaceholderText("e.g. CulmenDepth, FlipperLength")
        feat_layout.addWidget(self.feature_input)
        layout.addLayout(feat_layout)

        # --- Class Selection (comma-separated) ---
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Select 2 classes (comma separated):"))
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("e.g. Adelie, Chinstrap")
        class_layout.addWidget(self.class_input)
        layout.addLayout(class_layout)

        # --- Parameters ---
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout(params_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Learning Rate (η):"))
        self.learning_rate = QLineEdit()
        self.learning_rate.setPlaceholderText("e.g. 0.01")
        row1.addWidget(self.learning_rate)

        row1.addWidget(QLabel("Epochs (m):"))
        self.epochs = QLineEdit()
        self.epochs.setPlaceholderText("e.g. 10")
        row1.addWidget(self.epochs)

        params_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Threshold:"))
        self.threshold = QLineEdit()
        self.threshold.setPlaceholderText("e.g. 0 or 0.1")
        row2.addWidget(self.threshold)

        row2.addWidget(QLabel("Bias:"))
        self.bias_input = QLineEdit()
        self.bias_input.setPlaceholderText("Enter bias (default 0 for adaline and small random for perceptron)")
        row2.addWidget(self.bias_input)
        params_layout.addLayout(row2)

        layout.addWidget(params_group)

        # --- Algorithm Selection ---
        algo_group = QGroupBox("Algorithm Selection")
        algo_layout = QHBoxLayout(algo_group)
        self.perceptron_radio = QRadioButton("Perceptron")
        self.adaline_radio = QRadioButton("Adaline")
        self.perceptron_radio.setChecked(True)

        self.algo_button_group = QButtonGroup()
        self.algo_button_group.addButton(self.perceptron_radio)
        self.algo_button_group.addButton(self.adaline_radio)

        algo_layout.addWidget(self.perceptron_radio)
        algo_layout.addWidget(self.adaline_radio)
        layout.addWidget(algo_group)

        # --- Train Button ---
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self._train_model)
        layout.addWidget(self.train_button)

        # --- Last Error Display ---
        self.last_error_label = QLabel("Last Error: N/A")
        layout.addWidget(self.last_error_label)

        # --- Help / status area ---
        self.tuning_status = QTextEdit()
        self.tuning_status.setReadOnly(True)
        self.tuning_status.setFixedHeight(120)
        layout.addWidget(self.tuning_status)

        self.tabs.addTab(tuning_tab, "Model Tuning")

    # ==================================================
    # TAB 2: Prediction
    # ==================================================
    def _add_prediction_tab(self):
        prediction_tab = QWidget()
        layout = QVBoxLayout(prediction_tab)

        # layout.addWidget(QLabel("Enter values for a single sample (comma separated) or load test set via dataset:"))
        # self.predict_input = QLineEdit()
        # self.predict_input.setPlaceholderText("e.g. 0.45, 1.23  --- or leave empty to use x_test from preprocessing")
        # layout.addWidget(self.predict_input)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self._predict)
        layout.addWidget(self.predict_button)

        self.prediction_result = QTextEdit()
        self.prediction_result.setReadOnly(True)
        self.prediction_result.setFixedHeight(160)
        layout.addWidget(self.prediction_result)

        self.tabs.addTab(prediction_tab, "Prediction")

    # ==================================================
    # TAB 3: Evaluation
    # ==================================================
    def _add_evaluation_tab(self):
        evaluation_tab = QWidget()
        layout = QVBoxLayout(evaluation_tab)

        layout.addWidget(QLabel("Select Evaluation Plot:"))
        self.eval_plot_selector = QComboBox()
        self.eval_plot_selector.addItems([
            "Confusion Matrix",
            "Decision Boundary Plot"
        ])
        layout.addWidget(self.eval_plot_selector)

        self.show_plot_button = QPushButton("Show / Compute Evaluation")
        self.show_plot_button.clicked.connect(self._show_evaluation_plot)
        layout.addWidget(self.show_plot_button)

        self.accuracy_label = QLabel("Accuracy: N/A")
        layout.addWidget(self.accuracy_label)

        # --- Decision Boundary Canvas placeholder ---
        self.decision_canvas = MplCanvas(self)
        # self.decision_canvas.hide()  # hidden until user selects that option
        layout.addWidget(self.decision_canvas)

        # --- Evaluation text output ---
        self.eval_text = QTextEdit()
        self.eval_text.setReadOnly(True)
        self.eval_text.setFixedHeight(200)
        layout.addWidget(self.eval_text)

        self.tabs.addTab(evaluation_tab, "Evaluation")
    # ==================================================
    # EVENT HANDLERS
    # ==================================================
    def _validate_features_classes(self):
        features = [f.strip() for f in self.feature_input.text().split(",") if f.strip()]
        classes = [c.strip() for c in self.class_input.text().split(",") if c.strip()]

        if not features: features = ["CulmenDepth", "FlipperLength"]
        if not classes: classes = ["Adelie", "Chinstrap"]

        assert len(features) == 2, "Please select exactly two features."
        assert len(classes) == 2, "Please select exactly two classes."

        return features, classes

        
        

    def _load_dataset(self):
        """
        Opens file dialog, calls preprocess_data and stores returned data.
        Tries multiple unpacking patterns for flexibility.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset File",
            os.getcwd(),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            self.file_path_label.setText(os.path.basename(file_path))
            self.loaded_file = file_path

            self.features, self.classes = self._validate_features_classes()

            self.data, self.x_train_scaled, self.x_test_scaled, self.y_train, self.y_test, self.le = (
                preprocess_data(file_path, self.features, self.classes)
            )

            QMessageBox.information(self, "Dataset Loaded", "Dataset loaded and preprocessed successfully.")
            self.tuning_status.append(f"Loaded file: {os.path.basename(file_path)}")
            if self.features:
                self.tuning_status.append(f"Detected features: {self.features}")
            if self.classes:
                self.tuning_status.append(f"Detected classes: {self.classes}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{e}")

    def _train_model(self):
        self.eval_text.clear()

        if self.x_train_scaled is None or self.y_train is None:
            QMessageBox.warning(self, "Error", "Please load a dataset first.")
            return
        
        algo = "perceptron" if self.perceptron_radio.isChecked() else "adaline"
        _kwargs = {
            "perceptron": {"m": 10, "eta": 0.01, "th": 0},
            "adaline": {"m": 5, "eta": 0.005, "th": 0.02}
        }
        try:
            eta = float(self.learning_rate.text().strip() or _kwargs[algo]['eta'])
            epochs = int(self.epochs.text().strip() or _kwargs[algo]['m'])
            threshold = float(self.threshold.text().strip() or _kwargs[algo]['th'])
            bias = float(self.bias_input.text().strip() or 0)
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter numeric values for η, epochs, threshold, and bias.")
            return


        # Inform user
        self.tuning_status.append(f"Training {algo} (η={eta}, epochs={epochs}, threshold={threshold}, bias={bias}) ...")
        QApplication.processEvents()

        try:
            w, b = initialize_weights_and_bias(self.x_train_scaled, random=self.perceptron_radio.isChecked())
            b = b if bias == 0 else bias

            errors_or_mse, wb = train(
                self.x_train_scaled,
                self.y_train,
                w,
                b,
                algorithm=algo,
                epochs=epochs,
                learning_rate=eta,
                threshold=threshold
            )

            self.model_w, self.model_b = wb
            self.training_errors = errors_or_mse

            last_err = errors_or_mse[-1]
            self.last_error_label.setText(f"Last Error: {last_err}")
            self.tuning_status.append(f"Training complete. Last error: {last_err}")
        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))
            self.tuning_status.append(f"Training failed: {e}")

    def _predict(self):
        """
        Predicts using either:
        - a single sample input entered by user (comma separated)
        - or the x_test_scaled returned by preprocess_data
        """
        if self.model_w is None or self.model_b is None:
            QMessageBox.warning(self, "Error", "Model is not trained yet. Train the model first.")
            return

        try:
            if self.x_test_scaled is None:
                QMessageBox.warning(self, "Error", "No sample provided and no x_test available from preprocessing.")
                return
            
            linear_outputs = np.dot(self.x_test_scaled, self.model_w) + self.model_b
            preds = np.array([1 if signum(o) == 1 else 0 for o in linear_outputs])

            accuracy = np.mean(preds == self.y_test) * 100
            out_lines = [f"{i}: {a} -> {p}" for i, (p, a) in enumerate(zip(self.le.inverse_transform(preds[:-5]), self.data['Species']))]
            
            self.accuracy_label.setText(f"Accuracy: {accuracy:.2f}%")
            self.prediction_result.setPlainText(f"Actual vs Predictions (first {len(out_lines)}):\n" + "\n".join(out_lines))
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))

    def _show_evaluation_plot(self):
        selected_plot = self.eval_plot_selector.currentText()

        self.eval_text.clear()

        if selected_plot == "Decision Boundary Plot":
            self.decision_canvas.plot_decision_boundary(self.x_test_scaled, self.y_test, self.features, self.model_w, self.model_b)
            self.eval_text.setText("✅ Decision Boundary plotted successfully.")
        elif selected_plot == "Confusion Matrix":
            self.eval_text.setText("🧩 Confusion Matrix computation not yet implemented.")
        else:
            self.eval_text.setText("❌ Invalid plot selection.")
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NNTrainerUI()
    window.show()
    sys.exit(app.exec())
