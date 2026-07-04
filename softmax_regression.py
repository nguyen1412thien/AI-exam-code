import csv
import math
import os
import random


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=1000, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights = []
        self.bias = []
        self.classes = []
        self.loss_history = []

    def _softmax(self, scores):
        # Trừ giá trị lớn nhất để tránh exp bị quá lớn khi tính softmax.
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        total = sum(exp_scores)
        return [value / total for value in exp_scores]

    def _one_hot(self, label):
        # Chuyển nhãn thật thành vector one-hot để tính sai số từng lớp.
        return [1 if label == class_name else 0 for class_name in self.classes]

    def _predict_one_proba(self, sample):
        scores = []

        for class_index in range(len(self.classes)):
            score = self.bias[class_index]

            for feature_index in range(len(sample)):
                score += sample[feature_index] * self.weights[feature_index][class_index]

            scores.append(score)

        return self._softmax(scores)

    def _cross_entropy_loss(self, X, y):
        total_loss = 0
        eps = 1e-15

        for sample, label in zip(X, y):
            probabilities = self._predict_one_proba(sample)
            true_index = self.classes.index(label)

            # eps giúp log không nhận giá trị 0.
            total_loss += -math.log(probabilities[true_index] + eps)

        return total_loss / len(X)

    def fit(self, X, y):
        random.seed(self.random_state)
        self.classes = sorted(set(y))
        n_samples = len(X)
        n_features = len(X[0])
        n_classes = len(self.classes)

        # Khởi tạo trọng số nhỏ để mô hình bắt đầu học ổn định.
        self.weights = [
            [random.uniform(-0.01, 0.01) for _ in range(n_classes)]
            for _ in range(n_features)
        ]
        self.bias = [0.0 for _ in range(n_classes)]
        self.loss_history = []

        for epoch in range(self.epochs):
            dW = [[0.0 for _ in range(n_classes)] for _ in range(n_features)]
            db = [0.0 for _ in range(n_classes)]

            for sample, label in zip(X, y):
                probabilities = self._predict_one_proba(sample)
                target = self._one_hot(label)

                # Sai số softmax là xác suất dự đoán trừ nhãn one-hot.
                errors = [
                    probabilities[class_index] - target[class_index]
                    for class_index in range(n_classes)
                ]

                for feature_index in range(n_features):
                    for class_index in range(n_classes):
                        dW[feature_index][class_index] += (
                            sample[feature_index] * errors[class_index]
                        )

                for class_index in range(n_classes):
                    db[class_index] += errors[class_index]

            # Lấy trung bình gradient rồi cập nhật tham số bằng gradient descent.
            for feature_index in range(n_features):
                for class_index in range(n_classes):
                    self.weights[feature_index][class_index] -= (
                        self.learning_rate * dW[feature_index][class_index] / n_samples
                    )

            for class_index in range(n_classes):
                self.bias[class_index] -= self.learning_rate * db[class_index] / n_samples

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                loss = self._cross_entropy_loss(X, y)
                self.loss_history.append(loss)
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    def predict_proba(self, X):
        return [self._predict_one_proba(sample) for sample in X]

    def predict(self, X):
        predictions = []

        for probabilities in self.predict_proba(X):
            best_index = probabilities.index(max(probabilities))
            predictions.append(self.classes[best_index])

        return predictions

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for pred, true_label in zip(predictions, y) if pred == true_label)
        return correct / len(y)


def normalize_features(X):
    # Chuẩn hóa từng cột để các đặc trưng có cùng thang đo.
    n_features = len(X[0])
    means = []
    stds = []

    for feature_index in range(n_features):
        column = [sample[feature_index] for sample in X]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        std = math.sqrt(variance)

        means.append(mean)
        stds.append(std if std != 0 else 1)

    X_normalized = []
    for sample in X:
        new_sample = [
            (sample[feature_index] - means[feature_index]) / stds[feature_index]
            for feature_index in range(n_features)
        ]
        X_normalized.append(new_sample)

    return X_normalized


def train_test_split(X, y, test_size=0.3, random_state=42):
    random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)

    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = [X[index] for index in train_indices]
    y_train = [y[index] for index in train_indices]
    X_test = [X[index] for index in test_indices]
    y_test = [y[index] for index in test_indices]

    return X_train, X_test, y_train, y_test


def load_csv_dataset(file_path):
    X = []
    y = []

    with open(file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            X.append([float(row["x1"]), float(row["x2"])])
            y.append(row["label"])

    return X, y


def create_demo_dataset():
    # Hàm dự phòng khi muốn tự tạo nhanh dữ liệu minh họa.
    random.seed(7)
    X = []
    y = []
    centers = [
        ((1.0, 1.0), "Loai_A"),
        ((5.0, 1.5), "Loai_B"),
        ((3.0, 5.0), "Loai_C"),
    ]

    for center, label in centers:
        for _ in range(35):
            x1 = random.gauss(center[0], 0.45)
            x2 = random.gauss(center[1], 0.45)
            X.append([x1, x2])
            y.append(label)

    return X, y


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "softmax_demo.csv")

    X, y = load_csv_dataset(data_path)
    X = normalize_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = SoftmaxRegression(learning_rate=0.3, epochs=1000)
    model.fit(X_train, y_train)

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)

    print("\nKet qua danh gia:")
    print(f"Accuracy tren tap train: {train_acc:.2%}")
    print(f"Accuracy tren tap test : {test_acc:.2%}")

    sample = [X_test[0]]
    print("\nDu doan thu mot mau:")
    print("Nhan that:", y_test[0])
    print("Xac suat :", model.predict_proba(sample)[0])
    print("Du doan  :", model.predict(sample)[0])


if __name__ == "__main__":
    main()
