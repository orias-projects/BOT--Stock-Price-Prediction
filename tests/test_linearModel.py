# =====================================================================
# Kiểm thử mô hình tuyến tính — đối chứng của các mô hình cây
# =====================================================================
# Module này là mốc dưới của toàn bộ phần đánh giá, nên nó phải đúng
# chắc chắn: nếu baseline sai thì mọi kết luận "rừng hơn/kém đường
# thẳng" đều vô nghĩa.
#
# Thứ tự kiểm thử bám đúng chuỗi phụ thuộc của linearModel.py:
#
#   ① Chuẩn hoá      — thống kê cột, cột hằng số, tái dùng thống kê
#   ② Hồi quy tuyến tính — khôi phục được quan hệ tuyến tính đã biết
#   ③ Hồi quy logistic   — học được biên tuyến tính, giao diện đúng
#   ④ Ràng buộc          — báo lỗi rõ ràng khi dùng sai
#   ⑤ Đối chiếu chéo     — khớp scikit-learn trên cùng bài toán
# =====================================================================

import math
import random

import pytest

from libraries.linearModel import (LinearRegressor,
                                   LogisticRegressionClassifier,
                                   apply_standardization,
                                   calculate_column_statistics,
                                   count_linear_parameters)


@pytest.fixture
def linear_dataset():
    """y = 3·x₀ - 2·x₁ + 5, không nhiễu."""
    generator = random.Random(11)
    samples = [[generator.uniform(-3, 3), generator.uniform(-3, 3)]
               for _ in range(400)]
    targets = [3 * a - 2 * b + 5 for a, b in samples]
    return samples, targets


@pytest.fixture
def separable_dataset():
    """Hai lớp tách được bằng đường thẳng x₀ + x₁ = 0, nhãn -1 và 1."""
    generator = random.Random(23)
    samples, targets = [], []
    for _ in range(400):
        a, b = generator.uniform(-4, 4), generator.uniform(-4, 4)
        samples.append([a, b])
        targets.append(1 if a + b > 0 else -1)
    return samples, targets


# ---------------------------------------------------------------------
# ① Chuẩn hoá
# ---------------------------------------------------------------------
def test_column_statistics_match_manual_calculation():
    samples = [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]
    means, deviations = calculate_column_statistics(samples)

    assert means == pytest.approx([3.0, 20.0])
    assert deviations[0] == pytest.approx(math.sqrt(8 / 3))
    assert deviations[1] == pytest.approx(math.sqrt(200 / 3))


def test_constant_column_does_not_divide_by_zero():
    """Cột hằng số phải cho độ lệch 1.0 và chuẩn hoá thành toàn số 0."""
    samples = [[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]]
    means, deviations = calculate_column_statistics(samples)

    assert deviations[1] == 1.0
    scaled = apply_standardization(samples, means, deviations)
    assert all(row[1] == 0.0 for row in scaled)


def test_standardized_columns_have_zero_mean_and_unit_deviation():
    generator = random.Random(5)
    samples = [[generator.uniform(0, 100), generator.uniform(-1, 1)]
               for _ in range(200)]
    means, deviations = calculate_column_statistics(samples)
    scaled = apply_standardization(samples, means, deviations)

    scaled_means, scaled_deviations = calculate_column_statistics(scaled)
    assert scaled_means == pytest.approx([0.0, 0.0], abs=1e-9)
    assert scaled_deviations == pytest.approx([1.0, 1.0], abs=1e-9)


def test_prediction_reuses_training_statistics_not_new_ones(linear_dataset):
    """
    Thống kê chuẩn hoá phải là thống kê CỦA TẬP HUẤN LUYỆN.

    Nếu mô hình tính lại thống kê trên dữ liệu dự đoán thì dự đoán một
    mẫu duy nhất sẽ khác dự đoán chính mẫu đó nằm trong một lô — đây là
    một dạng rò rỉ tinh vi.
    """
    samples, targets = linear_dataset
    model = LinearRegressor(learning_rate=0.3, max_iterations=600).fit(
        samples, targets)

    batch_prediction = model.predict(samples[:20])[0]
    single_prediction = model.predict([samples[0]])[0]
    assert single_prediction == pytest.approx(batch_prediction)


# ---------------------------------------------------------------------
# ② Hồi quy tuyến tính
# ---------------------------------------------------------------------
def test_linear_regressor_recovers_known_relationship(linear_dataset):
    """
    Trên dữ liệu không nhiễu, mô hình phải khôi phục gần đúng quan hệ.

    Hệ số so sánh trên thang ĐÃ CHUẨN HOÁ, nên trọng số thật là
    3·σ₀ và -2·σ₁ chứ không phải 3 và -2.
    """
    samples, targets = linear_dataset
    model = LinearRegressor(learning_rate=0.3, max_iterations=2000).fit(
        samples, targets)

    _, deviations = calculate_column_statistics(samples)
    assert model.weights[0] == pytest.approx(3 * deviations[0], rel=0.02)
    assert model.weights[1] == pytest.approx(-2 * deviations[1], rel=0.02)

    predictions = model.predict(samples)
    errors = [abs(p - t) for p, t in zip(predictions, targets)]
    assert max(errors) < 0.5


def test_loss_decreases_monotonically(linear_dataset):
    samples, targets = linear_dataset
    model = LinearRegressor(learning_rate=0.1, max_iterations=200).fit(
        samples, targets)

    history = model.loss_history
    assert len(history) > 1
    assert all(later <= earlier + 1e-9
               for earlier, later in zip(history, history[1:]))


def test_l2_penalty_shrinks_weights(linear_dataset):
    samples, targets = linear_dataset
    plain = LinearRegressor(learning_rate=0.3, max_iterations=800,
                            l2_penalty=0.0).fit(samples, targets)
    penalised = LinearRegressor(learning_rate=0.3, max_iterations=800,
                                l2_penalty=0.5).fit(samples, targets)

    plain_size = sum(abs(weight) for weight in plain.weights)
    penalised_size = sum(abs(weight) for weight in penalised.weights)
    assert penalised_size < plain_size


def test_early_stopping_triggers_before_max_iterations(linear_dataset):
    samples, targets = linear_dataset
    model = LinearRegressor(learning_rate=0.5, max_iterations=5000,
                            tolerance=1e-6).fit(samples, targets)

    assert len(model.loss_history) < 5000


# ---------------------------------------------------------------------
# ③ Hồi quy logistic
# ---------------------------------------------------------------------
def test_logistic_learns_linear_boundary(separable_dataset):
    samples, targets = separable_dataset
    model = LogisticRegressionClassifier(learning_rate=0.5,
                                         max_iterations=1500).fit(samples, targets)

    predictions = model.predict(samples)
    accuracy = sum(1 for p, t in zip(predictions, targets) if p == t) / len(targets)
    assert accuracy > 0.95


def test_logistic_preserves_original_label_values(separable_dataset):
    """Nhãn -1 và 1 phải quay lại nguyên vẹn, không bị đổi thành 0 và 1."""
    samples, targets = separable_dataset
    model = LogisticRegressionClassifier(learning_rate=0.5,
                                         max_iterations=400).fit(samples, targets)

    assert model.label_space == [-1, 1]
    assert set(model.predict(samples)) <= {-1, 1}


def test_logistic_scores_stay_in_unit_interval(separable_dataset):
    samples, targets = separable_dataset
    model = LogisticRegressionClassifier(learning_rate=0.5,
                                         max_iterations=400).fit(samples, targets)

    scores = model.predict_scores(samples)
    assert all(0.0 <= score <= 1.0 for score in scores)

    probabilities = model.predict_probabilities(samples)
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in probabilities)


def test_logistic_threshold_shifts_decisions(separable_dataset):
    """Nâng ngưỡng phải làm số dự đoán dương giảm đi, không tăng lên."""
    samples, targets = separable_dataset
    model = LogisticRegressionClassifier(learning_rate=0.5,
                                         max_iterations=400).fit(samples, targets)

    counts = []
    for threshold in (0.2, 0.5, 0.8):
        model.threshold = threshold
        counts.append(sum(1 for label in model.predict(samples) if label == 1))

    assert counts[0] >= counts[1] >= counts[2]


def test_feature_weights_are_sorted_by_magnitude(separable_dataset):
    samples, targets = separable_dataset
    model = LogisticRegressionClassifier(learning_rate=0.5,
                                         max_iterations=400).fit(samples, targets)

    pairs = model.get_feature_weights(['alpha', 'beta'])
    assert [name for name, _ in pairs] == ['alpha', 'beta'] or \
           [name for name, _ in pairs] == ['beta', 'alpha']
    assert abs(pairs[0][1]) >= abs(pairs[1][1])


def test_parameter_count_includes_scaling_statistics(separable_dataset):
    """
    Thống kê chuẩn hoá PHẢI được tính là tham số.

    Không lưu chúng thì mô hình không dự đoán lại được, nên chúng là
    một phần của mô hình chứ không phải chi tiết cài đặt.
    """
    samples, targets = separable_dataset
    model = LogisticRegressionClassifier(max_iterations=50).fit(samples, targets)

    assert count_linear_parameters(model) == 2 + 1 + 2 * 2


# ---------------------------------------------------------------------
# ④ Ràng buộc — báo lỗi rõ ràng khi dùng sai
# ---------------------------------------------------------------------
def test_empty_training_set_is_rejected():
    with pytest.raises(ValueError, match="rỗng"):
        LinearRegressor().fit([], [])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="khác số nhãn"):
        LinearRegressor().fit([[1.0], [2.0]], [1.0])


def test_predicting_before_fitting_is_rejected():
    with pytest.raises(ValueError, match="chưa được huấn luyện"):
        LinearRegressor().predict([[1.0, 2.0]])


def test_logistic_rejects_more_than_two_classes():
    samples = [[float(index)] for index in range(9)]
    targets = [index % 3 for index in range(9)]

    with pytest.raises(ValueError, match="nhị phân"):
        LogisticRegressionClassifier().fit(samples, targets)


# ---------------------------------------------------------------------
# ⑤ Đối chiếu chéo với scikit-learn
# ---------------------------------------------------------------------
def test_logistic_agrees_with_scikit_learn(separable_dataset):
    """
    Cài đặt thuần Python phải cho cùng kết luận với thư viện chuẩn.

    Không đòi hệ số trùng khớp — hai bên dùng thuật toán tối ưu khác
    nhau — mà đòi HƯỚNG của biên quyết định và độ chính xác khớp nhau.
    """
    sklearn_linear = pytest.importorskip('sklearn.linear_model')

    samples, targets = separable_dataset
    own = LogisticRegressionClassifier(learning_rate=0.5,
                                       max_iterations=2000).fit(samples, targets)
    reference = sklearn_linear.LogisticRegression(max_iter=2000).fit(samples, targets)

    own_accuracy = sum(1 for p, t in zip(own.predict(samples), targets)
                       if p == t) / len(targets)
    reference_accuracy = reference.score(samples, targets)
    assert abs(own_accuracy - reference_accuracy) < 0.03

    own_ratio = own.weights[0] / own.weights[1]
    reference_ratio = reference.coef_[0][0] / reference.coef_[0][1]
    assert own_ratio == pytest.approx(reference_ratio, rel=0.15)
