# =====================================================================
# Linear Model — hồi quy tuyến tính và hồi quy logistic bằng Python thuần
# =====================================================================
# Module này tồn tại để làm ĐỐI CHỨNG cho các mô hình cây. Một mô hình
# rừng chỉ đáng dùng khi nó thắng được mô hình đơn giản nhất có thể học
# từ cùng bộ đặc trưng; nếu một đường thẳng đã đủ, cây là sự phức tạp
# không cần thiết.
#
# Toàn bộ tối ưu dùng gradient descent trên toàn tập, có chuẩn hoá đầu
# vào và phạt L2. Không dùng thư viện ngoài, cùng phong cách với
# randomForest.py để hai bên thay thế được cho nhau trong mã đánh giá.
#
# Thứ tự khai báo bám đúng chuỗi phụ thuộc:
#
#   ①  Thống kê cột      — trung bình và độ lệch chuẩn từng đặc trưng
#   ②  Chuẩn hoá         — dùng ①, biến mọi cột về cùng thang đo
#   ③  Tích vô hướng     — phép nhân nền của mọi dự đoán tuyến tính
#   ④  BaseLinearModel   — cần ①②③, chứa vòng lặp gradient descent
#   ⑤  LinearRegressor   — nhánh hồi quy, mất mát bình phương
#   ⑥  LogisticRegressionClassifier — nhánh phân loại, mất mát log
#   ⑦  Đếm tham số       — để so kích thước với mô hình rừng
#
# VÌ SAO PHẢI CHUẨN HOÁ (bước ②): gradient descent trên các cột lệch
# thang đo hàng trăm lần sẽ dao động hoặc hội tụ cực chậm. Cây quyết
# định miễn nhiễm với điều này vì nó chỉ so sánh thứ tự, còn mô hình
# tuyến tính thì không. Thống kê chuẩn hoá được học TRÊN TẬP HUẤN LUYỆN
# và lưu lại trong mô hình, nên lúc dự đoán không có rò rỉ.
# =====================================================================

import math

from libraries.rfMath import sigmoid, transpose_matrix


# ---------------------------------------------------------------------
# ① Thống kê cột — trung bình và độ lệch chuẩn của từng đặc trưng
# ---------------------------------------------------------------------
def calculate_column_statistics(samples):
    """
    Trung bình và độ lệch chuẩn của mỗi cột đặc trưng.

    Cột hằng số cho độ lệch chuẩn 0; ta trả về 1.0 thay vì 0 để bước
    chuẩn hoá không chia cho không. Cột đó sau khi chuẩn hoá thành toàn
    số 0, tức mô hình tự động bỏ qua nó — đúng như mong muốn, vì một cột
    không biến thiên thì không mang thông tin nào.

    Returns:
        (means, deviations) — hai danh sách cùng độ dài với số cột
    """
    if not samples:
        raise ValueError("Không thể tính thống kê trên tập rỗng.")

    columns = transpose_matrix(samples)
    means, deviations = [], []
    for column in columns:
        count = len(column)
        mean = sum(column) / count
        variance = sum((value - mean) ** 2 for value in column) / count
        deviation = math.sqrt(variance)
        means.append(mean)
        deviations.append(deviation if deviation > 1e-12 else 1.0)
    return means, deviations


# ---------------------------------------------------------------------
# ② Chuẩn hoá — dùng ① để đưa mọi đặc trưng về cùng thang đo
# ---------------------------------------------------------------------
def apply_standardization(samples, means, deviations):
    """
    Biến đổi z-score: (x - trung bình) / độ lệch chuẩn.

    Thống kê được TRUYỀN VÀO chứ không tính lại, để lúc dự đoán dùng
    đúng thống kê đã học từ tập huấn luyện.
    """
    return [[(value - mean) / deviation
             for value, mean, deviation in zip(sample, means, deviations)]
            for sample in samples]


# ---------------------------------------------------------------------
# ③ Tích vô hướng — phép nhân nền của mọi dự đoán tuyến tính
# ---------------------------------------------------------------------
def calculate_linear_output(sample, weights, bias):
    """Tổ hợp tuyến tính w·x + b."""
    return sum(value * weight for value, weight in zip(sample, weights)) + bias


# ---------------------------------------------------------------------
# ④ BaseLinearModel — cần ①②③, chứa vòng lặp gradient descent dùng chung
# ---------------------------------------------------------------------
class BaseLinearModel:
    """
    Khung chung cho hồi quy tuyến tính và hồi quy logistic.

    Hai lớp con chỉ khác nhau ở HÀM CHUYỂN ĐỔI đầu ra thô. Điều đáng
    chú ý: với cả mất mát bình phương lẫn mất mát log, gradient theo
    trọng số đều có dạng giống hệt nhau — Xᵀ(dự đoán - thực tế) / n.
    Nhờ vậy vòng lặp tối ưu viết một lần dùng cho cả hai.
    """

    def __init__(self, learning_rate=0.1, max_iterations=500,
                 l2_penalty=0.0, tolerance=1e-7, verbose=0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.l2_penalty = l2_penalty
        self.tolerance = tolerance
        self.verbose = verbose

        self.weights = []
        self.bias = 0.0
        self.feature_means = []
        self.feature_deviations = []
        self.loss_history = []

    # -- các lớp con định nghĩa hai phương thức dưới đây --------------
    def _transform_output(self, raw_output):
        raise NotImplementedError

    def _calculate_loss(self, predictions, targets):
        raise NotImplementedError

    def _prepare_targets(self, targets):
        """Chuyển nhãn thành số thực để tối ưu. Mặc định: giữ nguyên."""
        return [float(value) for value in targets]

    def fit(self, samples, targets):
        """
        Huấn luyện bằng gradient descent trên toàn tập.

        Dừng sớm khi mất mát gần như không đổi nữa (chênh lệch dưới
        tolerance) — tránh chạy hết số vòng vô ích khi đã hội tụ.
        """
        if not samples:
            raise ValueError("Không thể huấn luyện trên tập rỗng.")
        if len(samples) != len(targets):
            raise ValueError(
                f"Số mẫu ({len(samples)}) khác số nhãn ({len(targets)}).")

        self.feature_means, self.feature_deviations = \
            calculate_column_statistics(samples)
        scaled = apply_standardization(samples, self.feature_means,
                                       self.feature_deviations)
        numeric_targets = self._prepare_targets(targets)

        num_samples = len(scaled)
        num_features = len(scaled[0])
        self.weights = [0.0] * num_features
        self.bias = 0.0
        self.loss_history = []

        previous_loss = None
        for iteration in range(self.max_iterations):
            predictions = [self._transform_output(
                calculate_linear_output(sample, self.weights, self.bias))
                for sample in scaled]
            residuals = [prediction - target
                         for prediction, target in zip(predictions, numeric_targets)]

            weight_gradients = [0.0] * num_features
            for sample, residual in zip(scaled, residuals):
                for index, value in enumerate(sample):
                    weight_gradients[index] += value * residual
            weight_gradients = [
                gradient / num_samples + self.l2_penalty * weight
                for gradient, weight in zip(weight_gradients, self.weights)]
            bias_gradient = sum(residuals) / num_samples

            self.weights = [weight - self.learning_rate * gradient
                            for weight, gradient in zip(self.weights, weight_gradients)]
            self.bias -= self.learning_rate * bias_gradient

            loss = self._calculate_loss(predictions, numeric_targets)
            self.loss_history.append(loss)
            if self.verbose and iteration % max(1, self.max_iterations // 10) == 0:
                print(f"  vòng {iteration:4d} | mất mát = {loss:.6f}")
            if previous_loss is not None and abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss

        return self

    def _raw_outputs(self, samples):
        """Đầu ra thô w·x + b sau khi chuẩn hoá bằng thống kê đã học."""
        self._assert_fitted()
        scaled = apply_standardization(samples, self.feature_means,
                                       self.feature_deviations)
        return [calculate_linear_output(sample, self.weights, self.bias)
                for sample in scaled]

    def describe(self):
        """Tóm tắt mô hình để in ra hoặc ghi vào báo cáo."""
        self._assert_fitted()
        return {
            'num_features':    len(self.weights),
            'num_parameters':  count_linear_parameters(self),
            'num_iterations':  len(self.loss_history),
            'final_loss':      self.loss_history[-1] if self.loss_history else None,
            'learning_rate':   self.learning_rate,
            'l2_penalty':      self.l2_penalty,
        }

    def _assert_fitted(self):
        if not self.weights:
            raise ValueError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")


# ---------------------------------------------------------------------
# ⑤ LinearRegressor — nhánh hồi quy, đầu ra thô chính là dự đoán
# ---------------------------------------------------------------------
class LinearRegressor(BaseLinearModel):
    """
    Hồi quy tuyến tính có phạt L2, tối ưu bằng gradient descent.

    Dùng làm mốc dưới cho các mô hình hồi quy: nếu rừng hồi quy không
    thắng được mô hình này thì quan hệ trong dữ liệu vốn đã tuyến tính.
    """

    def _transform_output(self, raw_output):
        return raw_output

    def _calculate_loss(self, predictions, targets):
        """Mất mát bình phương trung bình."""
        return sum((prediction - target) ** 2
                   for prediction, target in zip(predictions, targets)) / len(targets)

    def predict(self, samples):
        return self._raw_outputs(samples)


# ---------------------------------------------------------------------
# ⑥ LogisticRegressionClassifier — nhánh phân loại, mất mát log
# ---------------------------------------------------------------------
class LogisticRegressionClassifier(BaseLinearModel):
    """
    Hồi quy logistic nhị phân.

    Giao diện khớp với RandomForestClassifier — fit / predict /
    predict_scores — nên cùng một đoạn mã đánh giá chạy được cho cả hai
    mà không cần rẽ nhánh.

    Nhãn KHÔNG bắt buộc phải là 0 và 1: lớp tự sắp xếp không gian nhãn
    và coi nhãn lớn hơn là lớp dương, giống quy ước của
    metricsClassification.
    """

    def __init__(self, learning_rate=0.1, max_iterations=500,
                 l2_penalty=0.0, tolerance=1e-7, threshold=0.5, verbose=0):
        super().__init__(learning_rate=learning_rate,
                         max_iterations=max_iterations,
                         l2_penalty=l2_penalty, tolerance=tolerance,
                         verbose=verbose)
        self.threshold = threshold
        self.label_space = []

    def _prepare_targets(self, targets):
        """Ánh xạ không gian nhãn về {0.0, 1.0} để tối ưu."""
        self.label_space = sorted(set(targets))
        if len(self.label_space) != 2:
            raise ValueError(
                f"Hồi quy logistic chỉ nhận bài toán nhị phân, "
                f"nhận được {len(self.label_space)} lớp: {self.label_space}")
        positive_label = self.label_space[-1]
        return [1.0 if value == positive_label else 0.0 for value in targets]

    def _transform_output(self, raw_output):
        return sigmoid(raw_output)

    def _calculate_loss(self, predictions, targets):
        """Mất mát log, có kẹp biên để tránh log(0)."""
        epsilon = 1e-15
        total = 0.0
        for prediction, target in zip(predictions, targets):
            clipped = min(max(prediction, epsilon), 1 - epsilon)
            total += -(target * math.log(clipped)
                       + (1 - target) * math.log(1 - clipped))
        return total / len(targets)

    def predict_scores(self, samples, positive_label=None):
        """
        Xác suất thuộc lớp dương — dùng cho ROC-AUC.

        Tham số positive_label có mặt để khớp giao diện của
        RandomForestClassifier, nhờ đó ReplayEngine và mã đánh giá gọi
        được cả hai loại mô hình mà không phải rẽ nhánh. Yêu cầu một
        lớp dương khác với lớp đã học sẽ đảo ngược điểm số.
        """
        scores = [sigmoid(output) for output in self._raw_outputs(samples)]
        if positive_label is not None and positive_label != self.label_space[-1]:
            if positive_label not in self.label_space:
                raise ValueError(
                    f"Nhãn {positive_label!r} không thuộc không gian nhãn "
                    f"đã học {self.label_space}.")
            return [1.0 - score for score in scores]
        return scores

    def predict_probabilities(self, samples):
        """Vector xác suất theo thứ tự label_space."""
        return [[1 - score, score] for score in self.predict_scores(samples)]

    def predict(self, samples):
        """Nhãn dự đoán, cắt tại threshold."""
        negative_label, positive_label = self.label_space
        return [positive_label if score >= self.threshold else negative_label
                for score in self.predict_scores(samples)]

    def get_feature_weights(self, feature_names=None):
        """
        Trọng số kèm tên đặc trưng, sắp theo độ lớn giảm dần.

        Vì đầu vào đã được chuẩn hoá, các trọng số so sánh trực tiếp
        được với nhau — đây là điều bảng trọng số của mô hình tuyến tính
        chưa chuẩn hoá KHÔNG làm được.
        """
        self._assert_fitted()
        names = feature_names or [f'feature_{index}'
                                  for index in range(len(self.weights))]
        pairs = list(zip(names, self.weights))
        return sorted(pairs, key=lambda pair: abs(pair[1]), reverse=True)


# ---------------------------------------------------------------------
# ⑦ Đếm tham số — để đặt cạnh số tham số của mô hình rừng
# ---------------------------------------------------------------------
def count_linear_parameters(model):
    """
    Số tham số của một mô hình tuyến tính đã huấn luyện.

    Gồm trọng số, hệ số chặn, và cặp thống kê chuẩn hoá mỗi cột — vì
    thống kê cũng phải lưu cùng mô hình thì lúc dự đoán mới tái tạo
    được đúng phép biến đổi.
    """
    return len(model.weights) + 1 + 2 * len(model.feature_means)
