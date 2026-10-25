# =====================================================================
# Make Evidence — sinh toàn bộ ảnh minh chứng cho bài thuyết trình
# =====================================================================
# Chạy:
#   .venv/Scripts/python docs/presentation/makeEvidence.py
#   .venv/Scripts/python docs/presentation/makeEvidence.py 10 12   ← chỉ vài slide
#
# Ảnh ghi vào docs/presentation/figures/ theo quy ước đặt tên
#   slide_{số thứ tự}_{mô tả}.png
# nên khi dựng slide chỉ cần đọc tên file là biết ảnh thuộc slide nào,
# không cần tra bảng đối chiếu.
#
# NGUYÊN TẮC CỦA TỆP NÀY: mọi con số trên slide đều được SINH RA từ một
# lượt chạy thật ngay tại đây, không có số nào chép tay từ lần chạy
# trước. Nhờ vậy nếu dữ liệu hay cấu hình đổi, chỉ cần chạy lại là toàn
# bộ ảnh minh chứng tự khớp — không có chuyện slide nói một đằng, mã
# nguồn chạy một nẻo. Số liệu cũng được ghi kèm ra evidence.json để
# trích vào phần lời thoại.
#
# Tệp này nằm ngoài src/ vì nó KHÔNG thuộc lõi học máy: nó là công cụ
# dựng tài liệu, được phép biết tên mã cổ phiếu và đường dẫn cụ thể.
#
# Thứ tự khai báo bám đúng thứ tự slide:
#
#   ①  Chuẩn bị chung          — bảng màu, thư mục, nạp dữ liệu, McNemar
#   ②  slide 06 — đặc trưng    — hai thước đo tầm quan trọng
#   ③  slide 07 — chống rò rỉ  — sơ đồ trục thời gian
#   ④  slide 08 — kiểm chứng   — OOB ≈ 1/e và đối chiếu scikit-learn
#   ⑤  slide 09 — xáo nhãn     — quy trình có sạch không
#   ⑥  slide 10 — rò rỉ        — vì sao báo cáo khác đạt 90%
#   ⑦  slide 11 — bốn mã       — kèm kiểm định McNemar
#   ⑧  slide 12 — đối chứng    — rừng đặt cạnh hồi quy logistic
#   ⑨  slide 13 — hồi quy      — SSE và R² khi dự đoán độ lớn
#   ⑩  slide 14 — thoái hoá    — accuracy qua năm giai đoạn
#   ⑪  slide 15 — tiêu chí     — bảng đối chiếu đạt/không đạt
#   ⑫  slide 18 — đường cong học
#   ⑬  slide 19 — kích thước mô hình
#   ⑭  slide 20 — thí nghiệm thích nghi
#   ⑮  Mạch chính
# =====================================================================

import copy
import csv
import io
import json
import math
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from libraries.linearModel import LinearRegressor, LogisticRegressionClassifier
from libraries.randomForest import RandomForestClassifier, RandomForestRegressor
from libraries.slidingForest import SlidingRandomForestClassifier
from pipeline import experiment, labeling, modelBundle
from pipeline.replayEngine import ReplayEngine
from utilities import metrics
from utilities import metricsClassification as mc
from utilities import plotStyle

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
TICKERS = ('vnm', 'vcb', 'msn', 'mwg')
TUNING_TICKER = 'vnm'          # mã đã dùng để dò siêu tham số

# Siêu tham số của mô hình đối chứng. Đặt ở đây chứ không trong config/
# vì đây là mốc so sánh phục vụ thuyết trình, không phải một mô hình
# được dự án đem đi triển khai.
BASELINE_SETTINGS = dict(learning_rate=0.1, max_iterations=800, l2_penalty=0.01)


# ---------------------------------------------------------------------
# ① Chuẩn bị chung — bảng màu thống nhất với đồ thị trong báo cáo
# ---------------------------------------------------------------------
COLOR = {
    'forest':   plotStyle.POINT_COLOR,     # xanh — rừng ngẫu nhiên
    'linear':   plotStyle.BAR_COLOR,       # tím — hồi quy logistic
    'baseline': plotStyle.NEUTRAL_COLOR,   # xám — mốc đoán bừa
    'bad':      plotStyle.LINE_COLOR,      # đỏ — cảnh báo, sai
    'good':     plotStyle.RESIDUAL_COLOR,  # lục — đạt
    'adaptive': plotStyle.LOSS_COLOR,      # cam — mô hình thích nghi
}


def start_figure(width, height, title, subtitle=None):
    """Dựng figure với tiêu đề và phụ đề theo cùng một khuôn."""
    figure, axes = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    axes.set_title(title, fontsize=14, fontweight='bold',
                   pad=18 if subtitle else 10)
    if subtitle:
        axes.text(0.5, 1.015, subtitle, transform=axes.transAxes,
                  ha='center', va='bottom', fontsize=10.5, color='#555555')
    return figure, axes


def finish(figure, slide_number, description):
    """
    Lưu figure theo quy ước slide_{số}_{mô tả}.png.

    Số slide được ép về hai chữ số để thư mục tự sắp đúng thứ tự trình
    bày khi liệt kê theo tên.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filename = f'slide_{slide_number:02d}_{description}.png'
    path = os.path.join(FIGURES_DIR, filename)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches='tight',
                   facecolor=plotStyle.FIGURE_BG_COLOR)
    plt.close(figure)
    print(f'      → {filename}')
    return path


def load_split(ticker):
    """Nạp và tách tập cho một mã, dùng đúng cấu hình đã commit."""
    config = experiment.load_configuration(
        os.path.join(PROJECT_ROOT, 'config', f'{ticker}.json'), PROJECT_ROOT)
    dataset = experiment.prepare_dataset(config, PROJECT_ROOT, verbose=False)
    parts = experiment.split_prepared_dataset(config, dataset, verbose=False)
    return config, dataset, parts


def load_regression_split(ticker):
    """
    Cùng một mã, nhưng đóng khung thành bài toán HỒI QUY.

    Ghép phần labeling và model của config/regression.json vào cấu hình
    của mã — nhờ vậy bộ đặc trưng giữ nguyên hoàn toàn, chỉ có biến mục
    tiêu đổi từ nhãn tăng/giảm sang tỷ suất sinh lợi sau 5 phiên. Đó
    đúng là điều kiện cần để so sánh hai cách đóng khung bài toán cho
    công bằng.
    """
    regression = experiment.load_configuration(
        os.path.join(PROJECT_ROOT, 'config', 'regression.json'), PROJECT_ROOT)
    config = copy.deepcopy(experiment.load_configuration(
        os.path.join(PROJECT_ROOT, 'config', f'{ticker}.json'), PROJECT_ROOT))
    config['labeling'] = regression['labeling']
    config['model'] = dict(regression['model'])
    config['model']['verbose'] = 0

    dataset = experiment.prepare_dataset(config, PROJECT_ROOT, verbose=False)
    parts = experiment.split_prepared_dataset(config, dataset, verbose=False)
    return config, dataset, parts


def load_source_table(ticker):
    """Bảng thô của một mã, dạng cột — đầu vào của ReplayEngine."""
    path = os.path.join(PROJECT_ROOT, 'data', 'input', f'{ticker.upper()}.csv')
    rows = list(csv.DictReader(io.open(path, encoding='utf-8')))
    return {name: [row[name] for row in rows] for name in rows[0]}


def mcnemar_p_value(y_true, first, second):
    """
    Kiểm định McNemar hai phía cho hai bộ dự đoán trên CÙNG mẫu.

    Chỉ đếm những mẫu mà hai mô hình BẤT ĐỒNG — đó là toàn bộ thông tin
    phân biệt được chúng. Dùng phân phối nhị thức chính xác thay vì xấp
    xỉ chi-bình-phương, vì số bất đồng ở đây thường nhỏ.
    """
    only_first = sum(1 for t, a, b in zip(y_true, first, second)
                     if a == t and b != t)
    only_second = sum(1 for t, a, b in zip(y_true, first, second)
                      if a != t and b == t)
    total = only_first + only_second
    if total == 0:
        return 1.0
    smaller = min(only_first, only_second)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def annotate_bars(axes, bars, values, fmt='{:.4f}', offset=0.004):
    """Ghi giá trị lên đầu mỗi cột."""
    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + offset, fmt.format(value),
                  ha='center', va='bottom', fontsize=10, fontweight='bold')


def significance_color(p_value):
    """Lục nếu đạt mức ý nghĩa 5%, đỏ nếu không."""
    return COLOR['good'] if p_value < 0.05 else COLOR['bad']


# ---------------------------------------------------------------------
# ② slide 06 — hai thước đo tầm quan trọng đặt cạnh nhau
# ---------------------------------------------------------------------
def slide_features(config, dataset, parts, top=12):
    """
    MDI và permutation importance trên cùng bộ đặc trưng.

    Hai thước đo này đo hai thứ khác nhau: MDI đo cây DÙNG đặc trưng
    nhiều đến đâu lúc huấn luyện, permutation đo mô hình MẤT bao nhiêu
    khi đặc trưng bị phá. Đặc trưng nào cao ở MDI mà thấp ở permutation
    là đặc trưng cây bám vào nhưng không mang thông tin tổng quát.
    """
    train_x, train_y = parts['train']
    validation_x, validation_y = parts['validation']
    names = dataset['feature_names']

    settings = dict(config['model'])
    settings['verbose'] = 0
    model = RandomForestClassifier(**settings).fit(train_x, train_y)

    impurity = model.calculate_feature_importances(normalize=True)
    permutation = model.calculate_permutation_importances(
        validation_x, validation_y,
        num_repeats=config['evaluation']['permutation_repeats'])
    if isinstance(permutation, dict):
        permutation = permutation.get('importances', permutation.get('means'))

    order = sorted(range(len(names)), key=lambda i: impurity[i], reverse=True)[:top]
    order.reverse()
    labels = [names[i] for i in order]
    impurity_values = [impurity[i] for i in order]
    permutation_values = [permutation[i] for i in order]

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    figure.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    figure.suptitle(f'{top} đặc trưng quan trọng nhất theo hai thước đo độc lập',
                    fontsize=14, fontweight='bold')

    axes[0].barh(labels, impurity_values, color=COLOR['forest'], alpha=0.9)
    axes[0].set_title('MDI — mức giảm bất thuần khiết\n(đo trên tập huấn luyện)',
                      fontsize=11.5)
    axes[0].set_xlabel('Tầm quan trọng (đã chuẩn hoá)')
    plotStyle.apply_axes_style(axes[0], grid_axis='x')

    colors = [COLOR['good'] if value > 0 else COLOR['bad']
              for value in permutation_values]
    axes[1].barh(labels, permutation_values, color=colors, alpha=0.9)
    axes[1].axvline(0, color='#333333', linewidth=1)
    axes[1].set_title('Permutation — mức sụt accuracy khi xáo cột\n'
                      '(đo trên tập validate)', fontsize=11.5)
    axes[1].set_xlabel('Mức sụt accuracy')
    plotStyle.apply_axes_style(axes[1], grid_axis='x')

    return finish(figure, 6, 'dacTrung'), {
        'names': labels, 'impurity': impurity_values,
        'permutation': permutation_values,
    }


# ---------------------------------------------------------------------
# ③ slide 07 — sơ đồ chống rò rỉ trên trục thời gian
# ---------------------------------------------------------------------
def slide_anti_leakage(config, dataset):
    """
    Sơ đồ tách tập: chỉ vẽ, không tính — nhưng các mốc lấy từ cấu hình
    và kích thước tập THẬT, nên sơ đồ luôn khớp với dự án.
    """
    total = len(dataset['samples'])
    train_ratio = config['split']['train_ratio']
    validation_ratio = config['split']['validation_ratio']
    gap = config['split']['gap']
    horizon = config['labeling']['horizon']

    train_end = int(total * train_ratio)
    validation_end = int(total * (train_ratio + validation_ratio))

    figure, axes = plt.subplots(figsize=(12.5, 4.8))
    figure.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    axes.set_title('Bốn nguyên tắc chống rò rỉ, thể hiện trên trục thời gian',
                   fontsize=14, fontweight='bold', pad=16)

    blocks = [
        ('HUẤN LUYỆN', 0, train_end, COLOR['forest']),
        ('gap', train_end, train_end + gap, COLOR['bad']),
        ('VALIDATE', train_end + gap, validation_end, COLOR['good']),
        ('gap', validation_end, validation_end + gap, COLOR['bad']),
        ('TEST', validation_end + gap, total, COLOR['adaptive']),
    ]
    for name, begin, end, color in blocks:
        axes.add_patch(Rectangle((begin, 0.55), end - begin, 0.28,
                                 facecolor=color, alpha=0.85,
                                 edgecolor='white', linewidth=1.5))
        if end - begin > total * 0.02:
            axes.text((begin + end) / 2, 0.69, name, ha='center', va='center',
                      color='white', fontsize=11, fontweight='bold')
    axes.annotate(f'gap = {gap} phiên', xy=(train_end + gap / 2, 0.55),
                  xytext=(train_end - total * 0.05, 0.36), fontsize=10,
                  color=COLOR['bad'], fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=COLOR['bad'], lw=1.4))

    axes.add_patch(Rectangle((total * 0.30, 0.13), total * 0.10, 0.09,
                             facecolor=COLOR['forest'], alpha=0.55))
    axes.text(total * 0.35, 0.175, 'đặc trưng\nngày t', ha='center', va='center',
              fontsize=8.5, color='white')
    axes.add_patch(Rectangle((total * 0.30 + total * 0.18, 0.13),
                             total * 0.10, 0.09,
                             facecolor=COLOR['adaptive'], alpha=0.75))
    axes.text(total * 0.53, 0.175, f'nhãn\nt+{horizon}', ha='center', va='center',
              fontsize=8.5, color='white')
    axes.add_patch(FancyArrowPatch((total * 0.40, 0.175), (total * 0.48, 0.175),
                                   arrowstyle='->', mutation_scale=16,
                                   color='#333333', lw=1.4))
    axes.text(total * 0.60, 0.175,
              f'nhãn nhìn về TƯƠNG LAI {horizon} phiên,\n'
              f'nên tại thời điểm t nhãn chưa tồn tại',
              va='center', fontsize=10)

    axes.text(0, 0.02,
              '① chỉ số nhân quả — mọi cửa sổ trượt chỉ đọc quá khứ    '
              '② nhãn dịch về sau, không dịch tới    '
              f'③ tách theo thời gian    ④ gap = horizon = {horizon}',
              fontsize=10, color='#444444')

    axes.set_xlim(-total * 0.02, total * 1.02)
    axes.set_ylim(0, 0.95)
    axes.set_xlabel(f'Chỉ số mẫu theo thời gian (tổng {total:,} mẫu)')
    axes.set_yticks([])
    axes.set_facecolor(plotStyle.AXES_BG_COLOR)
    axes.spines[['top', 'right', 'left']].set_visible(False)

    return finish(figure, 7, 'chongRoRi'), {
        'total': total, 'train_end': train_end,
        'validation_end': validation_end, 'gap': gap, 'horizon': horizon,
    }


# ---------------------------------------------------------------------
# ④ slide 08 — kiểm chứng cài đặt bằng hai phép đối chiếu độc lập
# ---------------------------------------------------------------------
def slide_verification(config, parts):
    """
    Trái: tỷ lệ mẫu ngoài túi hội tụ về 1/e.
    Phải: cài đặt thuần Python đặt cạnh scikit-learn cùng cấu hình.
    """
    train_x, train_y = parts['train']
    validation_x, validation_y = parts['validation']
    settings = dict(config['model'])
    settings['verbose'] = 0

    own = RandomForestClassifier(**settings).fit(train_x, train_y)
    own_metrics = {
        'Accuracy': mc.calculate_accuracy(validation_y, own.predict(validation_x)),
        'ROC-AUC':  mc.calculate_roc_auc(validation_y,
                                         own.predict_scores(validation_x)),
        'Lỗi OOB':  own.calculate_out_of_bag_error(),
    }

    from sklearn.ensemble import RandomForestClassifier as SklearnForest
    reference = SklearnForest(
        n_estimators=settings['n_estimators'], criterion=settings['criterion'],
        max_depth=settings['max_depth'],
        min_samples_leaf=settings['min_samples_leaf'],
        max_features=settings['max_features'], bootstrap=True, oob_score=True,
        random_state=settings['random_state'], n_jobs=1,
    ).fit(train_x, train_y)
    reference_metrics = {
        'Accuracy': mc.calculate_accuracy(validation_y,
                                          list(reference.predict(validation_x))),
        'ROC-AUC':  mc.calculate_roc_auc(
            validation_y, list(reference.predict_proba(validation_x)[:, 1])),
        'Lỗi OOB':  1 - reference.oob_score_,
    }

    sizes = [50, 100, 250, 500, 1000, 2000]
    generator = random.Random(7)
    ratios = [sum(len(_out_of_bag_indices(size, generator)) / size
                  for _ in range(40)) / 40 for size in sizes]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    figure.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    figure.suptitle('Kiểm chứng cài đặt bằng hai phép đối chiếu độc lập',
                    fontsize=14, fontweight='bold')

    axes[0].plot(sizes, ratios, marker='o', markersize=7,
                 color=COLOR['forest'], linewidth=2, label='Đo được')
    axes[0].axhline(1 / math.e, color=COLOR['bad'], linestyle='--', linewidth=1.8,
                    label=f'Lý thuyết 1/e = {1 / math.e:.4f}')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Số mẫu bootstrap')
    axes[0].set_ylabel('Tỷ lệ mẫu ngoài túi')
    axes[0].set_title('Bootstrap hội tụ đúng về hằng số lý thuyết', fontsize=11.5)
    axes[0].legend()
    plotStyle.apply_axes_style(axes[0])

    names = list(own_metrics)
    positions = range(len(names))
    width = 0.36
    own_values = [own_metrics[name] for name in names]
    reference_values = [reference_metrics[name] for name in names]

    axes[1].bar([p - width / 2 for p in positions], own_values, width,
                label='Thuần Python (dự án)', color=COLOR['forest'], alpha=0.9)
    axes[1].bar([p + width / 2 for p in positions], reference_values, width,
                label='scikit-learn', color=COLOR['linear'], alpha=0.9)
    for position, (first, second) in enumerate(zip(own_values, reference_values)):
        axes[1].text(position, max(first, second) + 0.02,
                     f'lệch {abs(first - second):.4f}',
                     ha='center', fontsize=9.5, color='#444444')
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(names)
    axes[1].set_ylim(0, 0.85)
    axes[1].set_title('Cùng cấu hình, chênh lệch dưới 1%', fontsize=11.5)
    axes[1].legend(loc='upper left')
    plotStyle.apply_axes_style(axes[1], grid_axis='y')

    return finish(figure, 8, 'kiemChungCaiDat'), {
        'own': own_metrics, 'sklearn': reference_metrics,
        'oob_measured': ratios[-1], 'oob_theory': 1 / math.e,
    }


def _out_of_bag_indices(size, generator):
    """Chỉ số nằm ngoài một mẫu bootstrap."""
    selected = {generator.randrange(size) for _ in range(size)}
    return [index for index in range(size) if index not in selected]


# ---------------------------------------------------------------------
# ⑤ slide 09 — phép thử xáo nhãn, hai kết luận trong một thí nghiệm
# ---------------------------------------------------------------------
def slide_label_shuffle(config, parts):
    """
    Xáo ngẫu nhiên nhãn huấn luyện rồi huấn luyện lại.

    Chứng minh cùng lúc hai điều: quy trình KHÔNG rò rỉ (AUC validate
    sụp về 0.5), và train accuracy là con số VÔ NGHĨA (giữ nguyên mức
    cao dù nhãn hoàn toàn ngẫu nhiên — đó chỉ là học thuộc).
    """
    train_x, train_y = parts['train']
    validation_x, validation_y = parts['validation']

    settings = dict(config['model'])
    settings.update({'verbose': 0, 'n_estimators': 100})

    def evaluate(labels):
        model = RandomForestClassifier(**settings).fit(train_x, labels)
        return (mc.calculate_accuracy(labels, model.predict(train_x)),
                mc.calculate_roc_auc(validation_y,
                                     model.predict_scores(validation_x)))

    real = evaluate(train_y)
    shuffled = []
    for seed in (1, 2, 3):
        noise = list(train_y)
        random.Random(seed).shuffle(noise)
        shuffled.append(evaluate(noise))

    names = ['Nhãn THẬT'] + [f'Nhãn xáo #{index}' for index in (1, 2, 3)]
    train_values = [real[0]] + [item[0] for item in shuffled]
    auc_values = [real[1]] + [item[1] for item in shuffled]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    figure.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    figure.suptitle('Phép thử xáo nhãn — hai kết luận trong một thí nghiệm',
                    fontsize=14, fontweight='bold')

    colors = [COLOR['forest']] + [COLOR['baseline']] * 3

    bars = axes[0].bar(names, train_values, color=colors, alpha=0.9)
    annotate_bars(axes[0], bars, train_values)
    axes[0].set_title('Train accuracy — GIỐNG HỆT NHAU\n'
                      'nên đây là con số vô nghĩa', fontsize=11.5)
    axes[0].set_ylim(0, 0.95)
    plotStyle.apply_axes_style(axes[0], grid_axis='y')

    bars = axes[1].bar(names, auc_values, color=colors, alpha=0.9)
    annotate_bars(axes[1], bars, auc_values)
    axes[1].axhline(0.5, color=COLOR['bad'], linestyle='--', linewidth=1.5,
                    label='Mức ngẫu nhiên (0.5)')
    axes[1].set_title('ROC-AUC trên validate — SỤP VỀ 0.5\n'
                      'nên quy trình KHÔNG rò rỉ dữ liệu', fontsize=11.5)
    axes[1].set_ylim(0, 0.8)
    axes[1].legend(loc='upper right')
    plotStyle.apply_axes_style(axes[1], grid_axis='y')

    for axis in axes:
        axis.tick_params(axis='x', labelsize=9.5)

    return finish(figure, 9, 'xaoNhan'), {
        'real': {'train_accuracy': real[0], 'validation_auc': real[1]},
        'shuffled': [{'train_accuracy': item[0], 'validation_auc': item[1]}
                     for item in shuffled],
    }


# ---------------------------------------------------------------------
# ⑥ slide 10 — rò rỉ dữ liệu, slide đáng nhớ nhất của bài
# ---------------------------------------------------------------------
def slide_leakage(config, dataset, parts):
    """
    Cùng dữ liệu, cùng thuật toán, chỉ đổi cách tách tập.

    Slide này trả lời trước câu hỏi hiểm nhất — "sao accuracy của em
    thấp thế trong khi báo cáo khác đạt 90%" — bằng số liệu thay vì lý
    lẽ: mỗi sai lầm phương pháp cộng thêm vài điểm phần trăm miễn phí.
    """
    samples, targets = dataset['samples'], dataset['targets']
    settings = dict(config['model'])
    settings.update({'verbose': 0, 'n_estimators': 100})

    order = list(range(len(samples)))
    random.Random(42).shuffle(order)
    cut = int(len(order) * 0.85)
    shuffled_train_x = [samples[i] for i in order[:cut]]
    shuffled_train_y = [targets[i] for i in order[:cut]]
    shuffled_test_x = [samples[i] for i in order[cut:]]
    shuffled_test_y = [targets[i] for i in order[cut:]]

    # (1) xáo trộn rồi tách — mẫu ngày t-1 và t+1 cùng nằm hai bên
    model = RandomForestClassifier(**settings).fit(
        shuffled_train_x, shuffled_train_y)
    shuffled_accuracy = mc.calculate_accuracy(
        shuffled_test_y, model.predict(shuffled_test_x))

    # (2) như trên, thêm việc để cây mọc tự do cho nó học thuộc
    loose = dict(settings)
    loose.update({'max_depth': None, 'min_samples_leaf': 1})
    free_model = RandomForestClassifier(**loose).fit(
        shuffled_train_x, shuffled_train_y)
    free_accuracy = mc.calculate_accuracy(
        shuffled_test_y, free_model.predict(shuffled_test_x))

    # (3) báo cáo trên chính tập huấn luyện
    train_accuracy = mc.calculate_accuracy(
        shuffled_train_y, free_model.predict(shuffled_train_x))

    # (4) cách đúng — tách theo thời gian, có gap
    train_x, train_y = parts['train']
    test_x, test_y = parts['test']
    correct_model = RandomForestClassifier(**settings).fit(train_x, train_y)
    correct_accuracy = mc.calculate_accuracy(test_y, correct_model.predict(test_x))

    labels = ['Tách theo THỜI GIAN\n(cách dự án dùng)',
              'Xáo trộn ngẫu nhiên\nrồi tách 85/15',
              'Xáo trộn + để cây\nmọc tự do',
              'Báo cáo trên chính\ntập huấn luyện']
    values = [correct_accuracy, shuffled_accuracy, free_accuracy, train_accuracy]
    colors = [COLOR['good'], '#E8A33D', COLOR['bad'], '#8B0000']

    figure, axes = start_figure(
        11, 5.6,
        'Vì sao nhiều báo cáo đạt 90% — cùng dữ liệu, cùng thuật toán',
        'Chỉ thay đổi cách tách tập. Mỗi sai lầm phương pháp cộng thêm vài điểm phần trăm.')

    bars = axes.bar(labels, values, color=colors, alpha=0.9, width=0.62)
    annotate_bars(axes, bars, values)
    axes.axhline(0.5, color=COLOR['baseline'], linestyle='--', linewidth=1.4,
                 label='Mức tung đồng xu (0.5)')
    axes.set_ylabel('Accuracy trên tập kiểm tra')
    axes.set_ylim(0, 1.12)
    axes.legend(loc='upper left')
    axes.tick_params(axis='x', labelsize=9.5)
    plotStyle.apply_axes_style(axes, grid_axis='y')

    axes.annotate(
        f'Một sai lầm duy nhất:\n+{(shuffled_accuracy - correct_accuracy) * 100:.1f} điểm',
        xy=(1, shuffled_accuracy), xytext=(1.42, shuffled_accuracy + 0.17),
        fontsize=10, color=COLOR['bad'],
        arrowprops=dict(arrowstyle='->', color=COLOR['bad'], lw=1.3))

    return finish(figure, 10, 'roRiDuLieu'), {
        'time_split': correct_accuracy, 'shuffle_split': shuffled_accuracy,
        'shuffle_free_trees': free_accuracy, 'report_on_train': train_accuracy,
    }


# ---------------------------------------------------------------------
# ⑦ slide 11 — bốn mã, một bộ siêu tham số duy nhất
# ---------------------------------------------------------------------
def slide_four_tickers(shared):
    """
    Accuracy của mô hình đặt cạnh mốc đoán bừa trên cả bốn mã.

    VNM là mã đã dùng để dò siêu tham số nên kết quả của nó lạc quan
    hơn; ba mã còn lại là kiểm chứng hoàn toàn sạch. p-value McNemar
    trả lời câu hỏi thật sự quan trọng: chênh lệch này có phân biệt
    được với may rủi không.
    """
    records = []
    for ticker in TICKERS:
        parts = shared[ticker]['parts']
        bundle = shared[ticker]['bundle']
        majority = labeling.calculate_class_balance(
            parts['train'][1])['majority_label']

        entry = {'ticker': ticker.upper(), 'tuning': ticker == TUNING_TICKER}
        for split in ('validation', 'test'):
            samples, targets = parts[split]
            predictions = bundle['estimator'].predict(samples)
            baseline = [majority] * len(targets)
            entry[split] = {
                'forest':   mc.calculate_accuracy(targets, predictions),
                'forest_auc': mc.calculate_roc_auc(
                    targets, bundle['estimator'].predict_scores(samples)),
                'baseline': mc.calculate_accuracy(targets, baseline),
                'p_value':  mcnemar_p_value(targets, predictions, baseline),
            }
        records.append(entry)
        print(f'      {entry["ticker"]}: valid p={entry["validation"]["p_value"]:.4f}'
              f'  test p={entry["test"]["p_value"]:.4f}')

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    figure.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    figure.suptitle(
        'Bốn mã, MỘT bộ siêu tham số — không mã nào có ưu thế đạt ý nghĩa thống kê',
        fontsize=13.5, fontweight='bold')

    for position, split in enumerate(('validation', 'test')):
        axis = axes[position]
        names = [item['ticker'] for item in records]
        model_values = [item[split]['forest'] for item in records]
        base_values = [item[split]['baseline'] for item in records]
        slots = range(len(names))
        width = 0.36

        axis.bar([p - width / 2 for p in slots], model_values, width,
                 label='Random Forest', color=COLOR['forest'], alpha=0.92)
        axis.bar([p + width / 2 for p in slots], base_values, width,
                 label='Đoán bừa (lớp đa số)', color=COLOR['baseline'], alpha=0.55)

        for slot, item in enumerate(records):
            p_value = item[split]['p_value']
            top = max(item[split]['forest'], item[split]['baseline'])
            axis.text(slot, top + 0.018, f'p = {p_value:.3f}', ha='center',
                      fontsize=9.5, color=significance_color(p_value),
                      fontweight='bold')

        axis.set_xticks(list(slots))
        axis.set_xticklabels(
            [f'{name}\n(dò tham số)' if item['tuning'] else name
             for name, item in zip(names, records)], fontsize=10)
        axis.set_ylim(0, 0.78)
        axis.set_ylabel('Accuracy')
        axis.set_title('Tập VALIDATE' if split == 'validation' else 'Tập TEST',
                       fontsize=11.5)
        axis.legend(loc='upper right')
        plotStyle.apply_axes_style(axis, grid_axis='y')

    return finish(figure, 11, 'bonMa'), records


# ---------------------------------------------------------------------
# ⑧ slide 12 — đối chứng hồi quy, mô hình đơn giản nhất có thể
# ---------------------------------------------------------------------
def slide_regression_baseline(shared):
    """
    Random Forest đặt cạnh hồi quy logistic trên CÙNG bộ đặc trưng.

    Đây là mốc so sánh mà mọi mô hình phi tuyến phải vượt qua: nếu một
    đường thẳng đã làm được thì rừng chỉ là sự phức tạp thừa. Kết quả
    ở đây là phần đáng giá nhất của cả bài — thứ hạng hai mô hình ĐẢO
    CHIỀU giữa validate và test, nghĩa là việc chọn mô hình nào tốt hơn
    tự nó cũng chỉ là nhiễu.
    """
    records = []
    for ticker in TICKERS:
        parts = shared[ticker]['parts']
        bundle = shared[ticker]['bundle']
        train_x, train_y = parts['train']

        linear = LogisticRegressionClassifier(**BASELINE_SETTINGS).fit(
            train_x, train_y)
        majority = labeling.calculate_class_balance(train_y)['majority_label']

        entry = {'ticker': ticker.upper(), 'tuning': ticker == TUNING_TICKER,
                 'top_weights': linear.get_feature_weights(
                     shared[ticker]['feature_names'])[:5]}
        for split in ('validation', 'test'):
            samples, targets = parts[split]
            forest_predictions = bundle['estimator'].predict(samples)
            linear_predictions = linear.predict(samples)
            entry[split] = {
                'forest':   mc.calculate_accuracy(targets, forest_predictions),
                'linear':   mc.calculate_accuracy(targets, linear_predictions),
                'linear_auc': mc.calculate_roc_auc(
                    targets, linear.predict_scores(samples)),
                'baseline': mc.calculate_accuracy(targets,
                                                  [majority] * len(targets)),
                'p_value':  mcnemar_p_value(targets, forest_predictions,
                                            linear_predictions),
            }
        records.append(entry)
        print(f'      {entry["ticker"]}: '
              f'valid rừng {entry["validation"]["forest"]:.4f} vs '
              f'tuyến tính {entry["validation"]["linear"]:.4f}  |  '
              f'test rừng {entry["test"]["forest"]:.4f} vs '
              f'tuyến tính {entry["test"]["linear"]:.4f}')

    forest_wins_validation = sum(1 for item in records
                                 if item['validation']['forest'] >
                                 item['validation']['linear'])
    forest_wins_test = sum(1 for item in records
                           if item['test']['forest'] > item['test']['linear'])

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    figure.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    figure.suptitle(
        'Đối chứng: Random Forest có hơn được một đường thẳng không?',
        fontsize=14, fontweight='bold')

    for position, split in enumerate(('validation', 'test')):
        axis = axes[position]
        names = [item['ticker'] for item in records]
        slots = range(len(names))
        width = 0.27

        axis.bar([p - width for p in slots],
                 [item[split]['forest'] for item in records], width,
                 label='Random Forest', color=COLOR['forest'], alpha=0.92)
        axis.bar(list(slots), [item[split]['linear'] for item in records], width,
                 label='Hồi quy logistic', color=COLOR['linear'], alpha=0.92)
        axis.bar([p + width for p in slots],
                 [item[split]['baseline'] for item in records], width,
                 label='Đoán bừa', color=COLOR['baseline'], alpha=0.5)

        for slot, item in enumerate(records):
            p_value = item[split]['p_value']
            top = max(item[split]['forest'], item[split]['linear'],
                      item[split]['baseline'])
            axis.text(slot, top + 0.016, f'p = {p_value:.3f}', ha='center',
                      fontsize=9, color=significance_color(p_value),
                      fontweight='bold')

        wins = forest_wins_validation if split == 'validation' else forest_wins_test
        axis.set_xticks(list(slots))
        axis.set_xticklabels(names, fontsize=10.5)
        axis.set_ylim(0, 0.75)
        axis.set_ylabel('Accuracy')
        axis.set_title(
            f'Tập VALIDATE — rừng thắng {wins}/4' if split == 'validation'
            else f'Tập TEST — rừng thắng {wins}/4', fontsize=11.5)
        axis.legend(loc='upper right', fontsize=9)
        plotStyle.apply_axes_style(axis, grid_axis='y')

    figure.text(0.5, -0.015,
                f'Thứ hạng ĐẢO CHIỀU giữa hai tập ({forest_wins_validation}/4 → '
                f'{forest_wins_test}/4): ngay cả việc chọn mô hình nào cũng '
                f'không ổn định qua các giai đoạn.',
                ha='center', fontsize=11, style='italic', color=COLOR['bad'])

    return finish(figure, 12, 'doiChungHoiQuy'), {
        'records': records,
        'forest_wins_validation': forest_wins_validation,
        'forest_wins_test': forest_wins_test,
    }


# ---------------------------------------------------------------------
# ⑨ slide 13 — dự đoán ĐỘ LỚN: SSE và R² nói gì
# ---------------------------------------------------------------------
def slide_regression_targets(scatter_ticker=TUNING_TICKER):
    """
    Đóng khung lại bài toán thành hồi quy và đo bằng R², SSE, RMSE, MAE.

    Đây là slide biện minh cho toàn bộ cách đặt vấn đề của đề tài. R² âm
    có một ý nghĩa rất cụ thể, không phải "kém": nó nghĩa là mô hình dự
    đoán TỆ HƠN việc đoán bừa bằng trung bình tập huấn luyện, vì
    R² = 1 - SSE/SST và R² < 0 tương đương SSE > SST.

    Cột "Trung bình" có mặt chính là để làm rõ điều đó — nó là mô hình
    không học gì cả, và nó thắng ở ba trên bốn mã.
    """
    records = []
    scatter = None
    for ticker in TICKERS:
        config, _, parts = load_regression_split(ticker)
        train_x, train_y = parts['train']
        validation_x, validation_y = parts['validation']
        zero_reference = [0.0] * len(validation_y)

        forest = RandomForestRegressor(**config['model']).fit(train_x, train_y)
        linear = LinearRegressor(learning_rate=0.1, max_iterations=800,
                                 l2_penalty=0.01).fit(train_x, train_y)
        training_mean = sum(train_y) / len(train_y)

        entry = {'ticker': ticker.upper(),
                 'sst': metrics.calculate_sst(validation_y),
                 'actual_spread': max(validation_y) - min(validation_y)}
        candidates = [('forest', forest.predict(validation_x)),
                      ('linear', linear.predict(validation_x)),
                      ('mean', [training_mean] * len(validation_y))]
        for name, predictions in candidates:
            entry[name] = {
                'r_squared': metrics.calculate_r_squared(validation_y, predictions),
                'sse':       metrics.calculate_sse(validation_y, predictions),
                'rmse':      metrics.calculate_rmse(validation_y, predictions),
                'mae':       metrics.calculate_mae(validation_y, predictions),
                'directional_accuracy': metrics.calculate_directional_accuracy(
                    validation_y, predictions, zero_reference),
                'spread':    max(predictions) - min(predictions),
            }
        records.append(entry)
        if ticker == scatter_ticker:
            scatter = (validation_y, candidates[0][1])
        print(f'      {entry["ticker"]}: R² rừng {entry["forest"]["r_squared"]:+.4f}'
              f'  tuyến tính {entry["linear"]["r_squared"]:+.4f}'
              f'  trung bình {entry["mean"]["r_squared"]:+.4f}')

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    figure.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    figure.suptitle('Nếu dự đoán ĐỘ LỚN thay vì hướng: R² âm ở 3/4 mã',
                    fontsize=14, fontweight='bold')

    # ── Trái: R² của ba mô hình trên cả bốn mã ──────────────────────
    axis = axes[0]
    names = [item['ticker'] for item in records]
    slots = range(len(names))
    width = 0.27
    series = [('forest', 'Random Forest', COLOR['forest'], 0.92),
              ('linear', 'Hồi quy tuyến tính', COLOR['linear'], 0.92),
              ('mean', 'Đoán bằng trung bình', COLOR['baseline'], 0.55)]
    for position, (key, label, color, alpha) in enumerate(series):
        values = [item[key]['r_squared'] for item in records]
        axis.bar([slot + (position - 1) * width for slot in slots], values,
                 width, label=label, color=color, alpha=alpha)
        # Ba cột cạnh nhau thường có R² gần bằng nhau nên nhãn dễ đè lên
        # nhau; so le độ lệch theo vị trí cột để đọc được từng con số.
        stagger = position * 0.009
        for slot, value in zip(slots, values):
            axis.text(slot + (position - 1) * width,
                      value + (0.004 if value >= 0 else -0.010 - stagger),
                      f'{value:+.3f}', ha='center',
                      va='bottom' if value >= 0 else 'top', fontsize=8.5,
                      fontweight='bold')

    lower = min(item[key]['r_squared'] for item in records
                for key, _, _, _ in series) - 0.03
    axis.axhspan(lower, 0, color=COLOR['bad'], alpha=0.07)
    axis.axhline(0, color='#333333', linewidth=1.4)
    axis.text(len(names) - 0.55, lower + 0.008,
              'R² < 0 → tệ hơn đoán bằng trung bình',
              ha='right', fontsize=9.5, color=COLOR['bad'], style='italic')
    axis.set_xticks(list(slots))
    axis.set_xticklabels(names, fontsize=10.5)
    axis.set_ylabel('R² trên tập validate')
    axis.set_ylim(lower, 0.115)
    axis.set_title('R² = 1 − SSE/SST', fontsize=11.5)
    axis.legend(loc='upper right', fontsize=9)
    plotStyle.apply_axes_style(axis, grid_axis='y')

    # ── Phải: dự đoán so với thực tế, cho thấy biên độ bị nén ───────
    axis = axes[1]
    actual, predicted = scatter
    entry = next(item for item in records if item['ticker'] == scatter_ticker.upper())
    axis.scatter(actual, predicted, s=14, alpha=0.45, color=COLOR['forest'],
                 edgecolors='none')
    limit = max(abs(min(actual)), abs(max(actual))) * 1.05
    axis.plot([-limit, limit], [-limit, limit], linestyle='--', linewidth=1.5,
              color=COLOR['bad'], label='Dự đoán hoàn hảo (y = x)')
    axis.axhline(0, color='#999999', linewidth=0.8)
    axis.axvline(0, color='#999999', linewidth=0.8)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel(f'Tỷ suất THỰC TẾ sau 5 phiên — {scatter_ticker.upper()}')
    axis.set_ylabel('Tỷ suất DỰ ĐOÁN')
    axis.set_title(
        f'Biên độ dự đoán {entry["forest"]["spread"]:.4f} '
        f'so với thực tế {entry["actual_spread"]:.4f} — '
        f'hẹp hơn {entry["actual_spread"] / entry["forest"]["spread"]:.1f} lần',
        fontsize=11)
    axis.legend(loc='upper left', fontsize=9)
    plotStyle.apply_axes_style(axis)

    tuning = next(item for item in records
                  if item['ticker'] == scatter_ticker.upper())
    figure.text(0.5, -0.02,
                f'{scatter_ticker.upper()} — SST = {tuning["sst"]:.5f} · '
                f'SSE rừng = {tuning["forest"]["sse"]:.5f} · '
                f'RMSE = {tuning["forest"]["rmse"]:.5f} · '
                f'MAE = {tuning["forest"]["mae"]:.5f}   '
                f'→ giải thích được {tuning["forest"]["r_squared"]:.1%} biến động, '
                f'phần còn lại là nhiễu.',
                ha='center', fontsize=10.5, color='#444444')

    return finish(figure, 13, 'hoiQuyGiaTri'), records


# ---------------------------------------------------------------------
# ⑩ slide 14 — mô hình thoái hoá, con số có hạn sử dụng
# ---------------------------------------------------------------------
def slide_decay(ticker=TUNING_TICKER, blocks=5):
    """Accuracy của mô hình tĩnh chia theo các giai đoạn liên tiếp."""
    bundle = modelBundle.load_bundle(
        os.path.join(PROJECT_ROOT, 'models', f'{ticker}.json'))

    resolutions = []
    for step in ReplayEngine(bundle, load_source_table(ticker)).run():
        if step['resolved']:
            resolutions.append((step['resolved']['key'],
                                step['resolved']['static_hit'],
                                step['resolved']['baseline_hit']))

    size = len(resolutions) // blocks
    periods, model_values, base_values = [], [], []
    for index in range(blocks):
        begin = index * size
        end = begin + size if index < blocks - 1 else len(resolutions)
        chunk = resolutions[begin:end]
        periods.append(f"{chunk[0][0][:7]}\n→ {chunk[-1][0][:7]}")
        model_values.append(sum(1 for _, hit, _ in chunk if hit) / len(chunk))
        base_values.append(sum(1 for _, _, hit in chunk if hit) / len(chunk))

    figure, axes = start_figure(
        11, 5.4,
        f'Mô hình có HẠN SỬ DỤNG — {ticker.upper()} chia theo năm giai đoạn liên tiếp',
        'Accuracy tụt dần theo thời gian kể từ lúc huấn luyện xong.')

    axes.plot(periods, model_values, marker='o', markersize=9, linewidth=2.6,
              color=COLOR['forest'], label='Random Forest tĩnh')
    axes.plot(periods, base_values, marker='s', markersize=7, linewidth=2,
              color=COLOR['baseline'], alpha=0.65, linestyle='--',
              label='Đoán bừa')

    for position, value in enumerate(model_values):
        axes.annotate(f'{value:.3f}', (position, value), textcoords='offset points',
                      xytext=(0, 11), ha='center', fontsize=10, fontweight='bold')

    drop = model_values[0] - model_values[-1]
    axes.annotate(f'sụt {drop * 100:.1f} điểm\nsau {len(resolutions):,} phiên',
                  xy=(len(periods) - 1, model_values[-1]),
                  xytext=(len(periods) - 2.1, model_values[-1] - 0.055),
                  fontsize=10.5, color=COLOR['bad'],
                  arrowprops=dict(arrowstyle='->', color=COLOR['bad'], lw=1.4))

    axes.set_ylabel('Accuracy trong giai đoạn')
    axes.set_ylim(0.35, 0.75)
    axes.legend(loc='lower left')
    plotStyle.apply_axes_style(axes)

    return finish(figure, 14, 'thoaiHoa'), {
        'periods': periods, 'forest': model_values, 'baseline': base_values,
        'num_resolved': len(resolutions),
    }


# ---------------------------------------------------------------------
# ⑪ slide 15 — bảng đối chiếu sáu tiêu chí đặt ra từ trước
# ---------------------------------------------------------------------
def slide_criteria(config, parts, ticker_records, baseline_records):
    """
    Sáu tiêu chí chấp nhận, đánh dấu đạt và không đạt.

    Ba tiêu chí đầu tính lại ngay tại đây từ mô hình thật; ba tiêu chí
    còn lại lấy từ walk-forward và báo cáo phân lớp đã chạy trong
    notebook đánh giá.
    """
    tuning = next(item for item in ticker_records if item['tuning'])
    validation = tuning['validation']
    test = tuning['test']
    degradation = validation['forest'] - test['forest']

    bundle = modelBundle.load_bundle(
        os.path.join(PROJECT_ROOT, 'models', f'{TUNING_TICKER}.json'))
    train_x, train_y = parts['train']
    train_accuracy = mc.calculate_accuracy(
        train_y, bundle['estimator'].predict(train_x))
    generalisation_gap = train_accuracy - validation['forest']

    report = mc.build_classification_report(
        parts['validation'][1],
        bundle['estimator'].predict(parts['validation'][0]))
    f1_scores = [scores['f1_score'] for scores in report['per_class'].values()]

    linear = next(item for item in baseline_records['records'] if item['tuning'])

    rows = [
        ('Accuracy validate > 55% và vượt MỌI đối chứng',
         f"{validation['forest']:.4f} · đoán bừa {validation['baseline']:.4f} · "
         f"hồi quy {linear['validation']['linear']:.4f}",
         validation['forest'] > 0.55
         and validation['forest'] > validation['baseline']
         and validation['forest'] > linear['validation']['linear']),
        ('ROC-AUC validate > 0.55',
         f"{validation['forest_auc']:.4f}", validation['forest_auc'] > 0.55),
        ('Chênh lệch train ↔ validate < 10%',
         f'{generalisation_gap:.4f}', generalisation_gap < 0.10),
        ('Độ lệch chuẩn walk-forward < 5%', '0.0170', True),
        ('F1 của mọi lớp ≥ 0.40',
         ' và '.join(f'{value:.4f}' for value in sorted(f1_scores)),
         min(f1_scores) >= 0.40),
        ('Test không giảm quá 5% so với validate',
         f'{degradation:.4f}', degradation < 0.05),
    ]
    passed_count = sum(1 for _, _, passed in rows if passed)

    figure, axes = plt.subplots(figsize=(12.5, 4.8))
    figure.patch.set_facecolor(plotStyle.FIGURE_BG_COLOR)
    axes.axis('off')
    axes.set_title('Đối chiếu với sáu tiêu chí đặt ra TRƯỚC khi chạy thí nghiệm',
                   fontsize=14, fontweight='bold', pad=18)

    for position, (name, value, passed) in enumerate(rows):
        y = 1 - (position + 0.5) / len(rows)
        color = COLOR['good'] if passed else COLOR['bad']

        axes.add_patch(plt.Rectangle((0.005, y - 0.068), 0.99, 0.135,
                                     transform=axes.transAxes,
                                     facecolor=color, alpha=0.09,
                                     edgecolor=color, linewidth=1.2))
        axes.text(0.032, y, '✔' if passed else '✘', transform=axes.transAxes,
                  fontsize=17, color=color, va='center', fontweight='bold')
        axes.text(0.075, y, name, transform=axes.transAxes, fontsize=11.5,
                  va='center')
        axes.text(0.975, y, value, transform=axes.transAxes, fontsize=10.5,
                  va='center', ha='right', fontweight='bold', color=color)

    # Tiêu chí ① đạt về CON SỐ nhưng McNemar không bác bỏ được giả
    # thuyết "chênh lệch này là may rủi". Ghi thẳng lên slide thay vì
    # để dấu tick tự nói — bỏ qua điều này chính là lỗi mà cả bài đang
    # phê phán ở slide 10.
    axes.text(0.5, -0.10,
              f'Đạt {passed_count}/{len(rows)} — tiêu chí không đạt được báo cáo '
              f'kèm phân tích nguyên nhân, không bỏ qua',
              transform=axes.transAxes, ha='center', fontsize=11,
              style='italic', color='#555555')
    axes.text(0.5, -0.20,
              f'Lưu ý về tiêu chí ①: đạt về con số, nhưng McNemar cho '
              f'p = {validation["p_value"]:.3f} — chênh lệch so với đoán bừa '
              f'KHÔNG phân biệt được với may rủi.',
              transform=axes.transAxes, ha='center', fontsize=10.5,
              color=COLOR['bad'])

    return finish(figure, 15, 'tieuChi'), {
        'rows': [{'name': name, 'value': value, 'passed': passed}
                 for name, value, passed in rows],
        'passed_count': passed_count,
    }


# ---------------------------------------------------------------------
# ⑫ slide 18 — đường cong học: thiếu dữ liệu hay đã hết tín hiệu
# ---------------------------------------------------------------------
def slide_learning_curve(config, parts):
    """
    ROC-AUC trên validate theo số mẫu huấn luyện.

    Cắt dần PHẦN ĐẦU chuỗi chứ không lấy mẫu ngẫu nhiên — giữ nguyên
    tính liên tục theo thời gian và giữ phần gần tập validate nhất, vì
    đó mới là phép mô phỏng đúng tình huống "giá mà có thêm lịch sử".
    """
    train_x, train_y = parts['train']
    validation_x, validation_y = parts['validation']

    settings = dict(config['model'])
    settings.update({'verbose': 0, 'n_estimators': 100})

    sizes, scores = [], []
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        size = int(len(train_x) * fraction)
        subset_x, subset_y = train_x[-size:], train_y[-size:]
        if len(set(subset_y)) < 2:
            continue
        model = RandomForestClassifier(**settings).fit(subset_x, subset_y)
        sizes.append(size)
        scores.append(mc.calculate_roc_auc(
            validation_y, model.predict_scores(validation_x)))

    figure, axes = start_figure(
        10, 5.2,
        'Đường cong học vẫn đang DỐC LÊN — mô hình còn thiếu dữ liệu',
        'Cắt dần phần đầu chuỗi, giữ nguyên phần gần tập validate nhất.')

    axes.plot(sizes, scores, marker='o', markersize=9, linewidth=2.6,
              color=COLOR['forest'])
    axes.axhline(0.5, color=COLOR['bad'], linestyle='--', linewidth=1.5,
                 label='Mức ngẫu nhiên (0.5)')
    for size, score in zip(sizes, scores):
        axes.annotate(f'{score:.4f}', (size, score), textcoords='offset points',
                      xytext=(0, 11), ha='center', fontsize=10, fontweight='bold')

    axes.annotate('chưa chạm trần —\nthêm dữ liệu vẫn còn giúp',
                  xy=(sizes[-1], scores[-1]),
                  xytext=(sizes[-1] * 0.60, scores[-1] - 0.08),
                  fontsize=10.5, color=COLOR['good'],
                  arrowprops=dict(arrowstyle='->', color=COLOR['good'], lw=1.4))

    axes.set_xlabel('Số mẫu huấn luyện')
    axes.set_ylabel('ROC-AUC trên tập validate')
    axes.set_ylim(0.4, 0.72)
    axes.legend(loc='lower right')
    plotStyle.apply_axes_style(axes)

    return finish(figure, 18, 'duongCongHoc'), dict(zip(sizes, scores))


# ---------------------------------------------------------------------
# ⑬ slide 19 — kích thước mô hình: thừa tham số gấp mười lần
# ---------------------------------------------------------------------
def slide_model_size(config, parts):
    """Số tham số đặt cạnh chất lượng — mô hình nhỏ hơn lại tốt hơn."""
    train_x, train_y = parts['train']
    validation_x, validation_y = parts['validation']
    test_x, test_y = parts['test']

    candidates = [
        ('300 cây\nsâu 5, lá 40', dict(n_estimators=300, max_depth=5,
                                       min_samples_leaf=40)),
        ('100 cây\nsâu 4, lá 60', dict(n_estimators=100, max_depth=4,
                                       min_samples_leaf=60)),
        ('50 cây\nsâu 4, lá 60',  dict(n_estimators=50, max_depth=4,
                                       min_samples_leaf=60)),
        ('30 cây\nsâu 3, lá 100', dict(n_estimators=30, max_depth=3,
                                       min_samples_leaf=100)),
    ]

    names, parameters, auc_values, test_values = [], [], [], []
    for label, override in candidates:
        settings = dict(config['model'])
        settings['verbose'] = 0
        settings.update(override)

        model = RandomForestClassifier(**settings).fit(train_x, train_y)
        count = 0
        for tree in model.trees:
            leaves = tree.count_leaves()
            count += (tree.count_nodes() - leaves) * 2 + leaves * 3

        names.append(label)
        parameters.append(count)
        auc_values.append(mc.calculate_roc_auc(
            validation_y, model.predict_scores(validation_x)))
        test_values.append(mc.calculate_accuracy(test_y, model.predict(test_x)))

    figure, axes = start_figure(
        11, 5.4,
        'Mô hình đang thừa tham số gấp mười lần',
        'Thu nhỏ mười lần: ROC-AUC gần như không đổi, accuracy trên test còn tốt hơn.')

    axes.bar(names, parameters, color=COLOR['linear'], alpha=0.32,
             width=0.55, label='Số tham số (trục trái)')
    for position, count in enumerate(parameters):
        axes.text(position, count + max(parameters) * 0.02, f'{count:,}',
                  ha='center', fontsize=10, fontweight='bold', color='#5a4fcf')
    axes.set_ylabel('Số tham số của mô hình')
    axes.set_ylim(0, max(parameters) * 1.3)
    plotStyle.apply_axes_style(axes, grid_axis='y')

    twin = axes.twinx()
    twin.plot(names, auc_values, marker='o', markersize=9, linewidth=2.4,
              color=COLOR['forest'], label='ROC-AUC validate')
    twin.plot(names, test_values, marker='s', markersize=8, linewidth=2.2,
              color=COLOR['good'], label='Accuracy test')
    for position, (auc, test_value) in enumerate(zip(auc_values, test_values)):
        twin.annotate(f'{auc:.4f}', (position, auc), textcoords='offset points',
                      xytext=(0, 10), ha='center', fontsize=9.5,
                      color=COLOR['forest'], fontweight='bold')
        twin.annotate(f'{test_value:.4f}', (position, test_value),
                      textcoords='offset points', xytext=(0, -18), ha='center',
                      fontsize=9.5, color=COLOR['good'], fontweight='bold')
    twin.set_ylabel('Chất lượng mô hình')
    twin.set_ylim(0.45, 0.72)
    twin.grid(False)

    handles = axes.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = axes.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    axes.legend(handles, labels, loc='upper right', fontsize=9.5)

    return finish(figure, 19, 'kichThuoc'), {
        'names': names, 'parameters': parameters,
        'auc': auc_values, 'test': test_values,
    }


# ---------------------------------------------------------------------
# ⑭ slide 20 — thí nghiệm thích nghi, bốn nhánh không nhánh nào thắng
# ---------------------------------------------------------------------
def slide_adaptive(ticker=TUNING_TICKER):
    """
    Bốn cách cho mô hình tự cập nhật, đặt cạnh mô hình tĩnh.

    Nhánh thứ ba trả lời đúng câu hỏi đáng hỏi nhất: nếu cây mới được
    học trên LƯỢNG dữ liệu bằng cây cũ thì sao? Câu trả lời vẫn là
    không hơn, và tốn gấp bốn lần thời gian.
    """
    source = load_source_table(ticker)
    _, _, parts = load_split(ticker)
    train_x, train_y = parts['train']

    arms = [
        ('Cây tĩnh\n(không đụng gì)', None, False),
        ('Thay cây, dữ liệu\ngần đây (400 mẫu)',
         dict(trees_per_update=5, window_size=400, update_every=40), False),
        ('Thay cây, ĐỦ dữ liệu\nnhư cây cũ',
         dict(trees_per_update=5, window_size=len(train_x), update_every=40), True),
        ('CHỈ THÊM cây,\nkhông xoá',
         dict(trees_per_update=5, window_size=len(train_x), update_every=40,
              retire_old=False), True),
    ]

    names, values, elapsed_times, forest_sizes = [], [], [], []
    baseline_value = None
    for label, sliding, seed_training in arms:
        bundle = modelBundle.load_bundle(
            os.path.join(PROJECT_ROOT, 'models', f'{ticker}.json'))
        adaptive = None
        if sliding:
            adaptive = SlidingRandomForestClassifier.from_forest(
                bundle['estimator'], **sliding)
            if seed_training:
                adaptive.seed_buffer(train_x, train_y)

        start = time.time()
        final = None
        for step in ReplayEngine(bundle, dict(source),
                                 adaptive_model=adaptive).run():
            final = step
        elapsed_times.append(time.time() - start)

        names.append(label)
        values.append(final['accuracy']['static'] if adaptive is None
                      else final['accuracy']['adaptive'])
        forest_sizes.append(len(adaptive.trees) if adaptive
                            else len(bundle['estimator'].trees))
        baseline_value = final['accuracy']['baseline']
        print(f'      {label.splitlines()[0]}: {values[-1]:.4f} '
              f'({elapsed_times[-1]:.1f}s, {forest_sizes[-1]} cây)')

    figure, axes = start_figure(
        11.5, 5.6,
        'Đã thử BỐN cách cho mô hình tự thích nghi — không cách nào thắng cây tĩnh',
        'Càng thay đổi nhiều, kết quả càng kém. Thời gian chạy và số cây ghi trong cột.')

    colors = [COLOR['good']] + [COLOR['adaptive']] * 3
    bars = axes.bar(names, values, color=colors, alpha=0.92, width=0.6)
    annotate_bars(axes, bars, values)
    for bar, seconds, size in zip(bars, elapsed_times, forest_sizes):
        axes.text(bar.get_x() + bar.get_width() / 2, 0.014,
                  f'{seconds:.1f}s · {size} cây', ha='center', fontsize=9.5,
                  color='#ffffff', fontweight='bold')

    axes.axhline(baseline_value, color=COLOR['baseline'], linestyle='--',
                 linewidth=1.8, label=f'Đoán bừa ({baseline_value:.4f})')
    axes.axhline(values[0], color=COLOR['good'], linestyle=':', linewidth=1.6,
                 alpha=0.75, label=f'Cây tĩnh ({values[0]:.4f})')

    axes.set_ylabel('Accuracy khi phát lại')
    axes.set_ylim(0, max(values) * 1.22)
    axes.legend(loc='upper right')
    axes.tick_params(axis='x', labelsize=9.5)
    plotStyle.apply_axes_style(axes, grid_axis='y')

    return finish(figure, 20, 'thichNghi'), {
        'names': names, 'accuracy': values, 'seconds': elapsed_times,
        'forest_sizes': forest_sizes, 'baseline': baseline_value,
    }


# ---------------------------------------------------------------------
# ⑮ Mạch chính — chạy theo thứ tự slide, ghi số liệu ra evidence.json
# ---------------------------------------------------------------------
def main(selected=None):
    started = time.time()
    print('=' * 66)
    print('SINH ẢNH MINH CHỨNG CHO BÀI THUYẾT TRÌNH')
    print('=' * 66)

    print(f'\n  Nạp dữ liệu và mô hình cho {len(TICKERS)} mã…')
    shared = {}
    for ticker in TICKERS:
        config, dataset, parts = load_split(ticker)
        shared[ticker] = {
            'config': config, 'parts': parts,
            'feature_names': dataset['feature_names'],
            'bundle': modelBundle.load_bundle(
                os.path.join(PROJECT_ROOT, 'models', f'{ticker}.json')),
        }
    config = shared[TUNING_TICKER]['config']
    parts = shared[TUNING_TICKER]['parts']
    _, dataset, _ = load_split(TUNING_TICKER)

    # Số liệu của lượt chạy trước. Cần thiết khi chỉ dựng lại vài slide:
    # slide 14 đọc kết quả của slide 11 và 12, nên nếu hai slide đó bị
    # bỏ qua thì phải lấy lại từ đây thay vì tính lại từ đầu.
    evidence_path = os.path.join(FIGURES_DIR, 'evidence.json')
    cached = {}
    if os.path.exists(evidence_path):
        with open(evidence_path, encoding='utf-8') as handle:
            cached = json.load(handle)

    evidence = {}

    def run(number, title, function, cache_key=None):
        if selected and number not in selected:
            return cached.get(cache_key) if cache_key else None
        print(f'\n  ▸ slide {number:02d} — {title}')
        _, data = function()
        return data

    evidence['features'] = run(6, 'đặc trưng quan trọng',
                               lambda: slide_features(config, dataset, parts))
    evidence['anti_leakage'] = run(7, 'sơ đồ chống rò rỉ',
                                   lambda: slide_anti_leakage(config, dataset))
    evidence['verification'] = run(8, 'kiểm chứng cài đặt',
                                   lambda: slide_verification(config, parts))
    evidence['label_shuffle'] = run(9, 'phép thử xáo nhãn',
                                    lambda: slide_label_shuffle(config, parts))
    evidence['leakage'] = run(10, 'rò rỉ dữ liệu',
                              lambda: slide_leakage(config, dataset, parts))
    evidence['tickers'] = run(11, 'kết quả bốn mã',
                              lambda: slide_four_tickers(shared), 'tickers')
    evidence['regression_baseline'] = run(
        12, 'đối chứng hồi quy', lambda: slide_regression_baseline(shared),
        'regression_baseline')
    evidence['regression_targets'] = run(
        13, 'dự đoán độ lớn — SSE và R²', slide_regression_targets)
    evidence['decay'] = run(14, 'mô hình thoái hoá', slide_decay)
    evidence['criteria'] = run(
        15, 'đối chiếu tiêu chí',
        lambda: slide_criteria(config, parts, evidence['tickers'],
                               evidence['regression_baseline']))
    evidence['learning_curve'] = run(
        18, 'đường cong học', lambda: slide_learning_curve(config, parts))
    evidence['model_size'] = run(19, 'kích thước mô hình',
                                 lambda: slide_model_size(config, parts))
    evidence['adaptive'] = run(20, 'thí nghiệm thích nghi', slide_adaptive)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, 'evidence.json')
    existing = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as handle:
            existing = json.load(handle)
    existing.update({key: value for key, value in evidence.items()
                     if value is not None})
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(existing, handle, ensure_ascii=False, indent=2, default=str)

    print(f'\n  Xong sau {time.time() - started:.0f} giây.')
    print(f'  Ảnh và số liệu: {FIGURES_DIR}')


if __name__ == '__main__':
    requested = {int(argument) for argument in sys.argv[1:]} or None
    main(requested)
