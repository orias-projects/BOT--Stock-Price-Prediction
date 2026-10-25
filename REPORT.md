# BÁO CÁO ĐỒ ÁN

## DỰ ĐOÁN XU HƯỚNG GIÁ CỔ PHIẾU BẰNG RANDOM FOREST TỰ CÀI ĐẶT

> *Một nghiên cứu về quy trình học máy trung thực trên dữ liệu tài chính*

| | |
| --- | --- |
| **Học phần** | Machine Learning — Mini Project |
| **Đề tài** | Dự đoán xu hướng tăng/giảm giá cổ phiếu bằng Random Forest kết hợp phân tích kỹ thuật |
| **Mã nguồn** | `BOT--Machine-Learning` — Random Forest, Gradient Boosting, hồi quy tuyến tính/logistic cài đặt thuần Python |
| **Dữ liệu** | 4 mã HOSE: VNM, VCB, MSN, MWG · 2016-08-22 → 2026-08-21 |
| **Quy mô mã nguồn** | ~12 700 dòng Python · 222 kiểm thử tự động (toàn bộ ĐẠT) |
| **Khả năng tái lập** | `random_state = 42`, mọi con số trong báo cáo tái lập được từ mã nguồn |

---

## MỤC LỤC

1. [Tóm tắt](#1-tóm-tắt)
2. [Giới thiệu và mô tả bài toán thực tế](#2-giới-thiệu-và-mô-tả-bài-toán-thực-tế)
3. [Dữ liệu](#3-dữ-liệu)
4. [Phương pháp học máy](#4-phương-pháp-học-máy)
5. [Kết quả thí nghiệm](#5-kết-quả-thí-nghiệm)
6. [Chức năng chính của hệ thống và cách sử dụng](#6-chức-năng-chính-của-hệ-thống-và-cách-sử-dụng)
7. [Khó khăn gặp phải và cách khắc phục](#7-khó-khăn-gặp-phải-và-cách-khắc-phục)
8. [Tranh luận, khám phá và kết luận](#8-tranh-luận-khám-phá-và-kết-luận)
9. [Hướng phát triển trong tương lai](#9-hướng-phát-triển-trong-tương-lai)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)
11. [Phụ lục](#11-phụ-lục)

---

# 1. TÓM TẮT

Đồ án xây dựng một hệ thống học máy hoàn chỉnh — từ nạp dữ liệu thô đến máy chủ dự
đoán chạy trong Docker — để trả lời một câu hỏi cụ thể: **cho dữ liệu giá đến hết
phiên hôm nay, sau 5 phiên nữa giá đóng cửa sẽ cao hơn hay thấp hơn?**

Toàn bộ thuật toán (cây quyết định CART, Random Forest, Gradient Boosting, hồi quy
tuyến tính và logistic) được cài đặt **thuần Python, không dùng thư viện học máy**;
`scikit-learn` chỉ xuất hiện ở tầng đối chiếu để kiểm chứng cài đặt.

**Kết quả chính trên mã VNM:** Accuracy validate **0.6087**, ROC-AUC **0.6476**,
vượt mọi mốc đối chứng và thoả **5/6** tiêu chí chấp nhận đã ghi vào tài liệu
*trước khi* chạy thí nghiệm đầu tiên. Trên tập test — giai đoạn thị trường mô hình
chưa từng thấy — Accuracy giảm còn **0.5237**, và kiểm định McNemar cho
`p = 0.289`, tức **ưu thế so với đoán bừa chưa đạt ý nghĩa thống kê**.

Đóng góp thực chất của đồ án **không** nằm ở con số accuracy, mà ở bộ bằng chứng
cho thấy con số đó là trung thực:

| Bằng chứng | Kết quả |
| --- | --- |
| Phép thử xáo nhãn (kiểm tra rò rỉ) | AUC validate sụp từ 0.6418 xuống 0.41–0.54 |
| Tái hiện rò rỉ có kiểm soát | Cùng dữ liệu, chỉ đổi cách tách tập: 0.5292 → 0.5718 → 0.6612 → 1.0000 |
| Đối chiếu với `scikit-learn` | Chênh lệch < 1% trên cả 4 chỉ số; 4 đặc trưng đầu bảng trùng thứ hạng tuyệt đối |
| Tỷ lệ out-of-bag ↔ lý thuyết | Đo 0.3676 · lý thuyết `(1−1/n)ⁿ = 0.3676` · giới hạn `1/e = 0.3679` |
| Kiểm thử tự động | 222 test, toàn bộ ĐẠT |

**Kết luận một dòng:** hệ thống **chứng minh được quy trình**, **chưa chứng minh
được lợi nhuận** — và đồ án chủ động báo cáo cả hai vế thay vì chỉ vế thứ nhất.

---

# 2. GIỚI THIỆU VÀ MÔ TẢ BÀI TOÁN THỰC TẾ

## 2.1. Bối cảnh thực tế

Nhà đầu tư cá nhân trên thị trường chứng khoán Việt Nam ra quyết định mua/bán chủ
yếu dựa vào **phân tích kỹ thuật** — đọc các chỉ báo như RSI, MACD, Bollinger Bands
để phán đoán xu hướng phiên tới. Cách làm này có ba điểm yếu cố hữu:

1. **Chủ quan.** Hai người nhìn cùng một đồ thị RSI có thể ra hai kết luận trái
   ngược. Không có tiêu chí định lượng nào để nói ai đúng.
2. **Không đo được.** Không ai ghi lại tỷ lệ dự đoán đúng của chính mình một cách
   có hệ thống, nên không biết phương pháp đang dùng có hơn tung đồng xu hay không.
3. **Thiên lệch trí nhớ.** Người ta nhớ những lần dự đoán đúng và quên những lần
   sai — dẫn tới tự tin quá mức vào một phương pháp thực chất không hiệu quả.

Học máy giải quyết đúng ba điểm này: mô hình cho ra một quy tắc **cố định và tái
lập được**, hiệu năng của nó **đo được bằng số**, và quan trọng nhất — có thể so
sánh nó với **mốc đối chứng tầm thường** (đoán bừa) để biết phần đóng góp thật sự
là bao nhiêu.

## 2.2. Phát biểu bài toán

Cho chuỗi dữ liệu giá theo ngày `Date, Open, High, Low, Close, Volume`, đồ án giải
**hai bài toán song song trên cùng một tập đặc trưng**:

| Nhánh | Bài toán | Biến mục tiêu | Kiểu học |
| --- | --- | --- | --- |
| **A — chính** | Dự đoán **chiều** biến động sau `h` phiên | `y = 1` nếu `Close(t+h) > Close(t)`, ngược lại `y = 0` | Phân loại nhị phân |
| **B — phụ** | Dự đoán **độ lớn** biến động sau `h` phiên | `y = (Close(t+h) − Close(t)) / Close(t)` | Hồi quy |

Nhánh B không phải phần trang trí. Nó tồn tại để trả lời một câu hỏi phương pháp
luận: *bài toán này nên được đóng khung thành phân loại hay hồi quy?* Kết quả ở
[§5.7](#57-nhánh-b--dự-đoán-độ-lớn-biến-động) cho câu trả lời dứt khoát bằng số
liệu, chứ không bằng lập luận.

**Tầm nhìn dự báo `h` là tham số, không phải hằng số.** Cấu hình mặc định dùng
`h = 5` (một tuần giao dịch). Lý do chọn giá trị này được trình bày ở
[§4.2](#42-lựa-chọn-tầm-nhìn-dự-báo) — dựa trên thực nghiệm, không phải quy ước.

## 2.3. Đầu vào và đầu ra của hệ thống

```text
ĐẦU VÀO                          HỆ THỐNG                        ĐẦU RA
─────────                        ────────                        ──────
File OHLCV        ┌─────────────────────────────────┐   • Nhãn: TĂNG / GIẢM
(.csv/.xlsx/.json)│ Tiền xử lý → 27 chỉ báo kỹ thuật│   • Xác suất tăng ∈ [0,1]
≥ 50 phiên     ──►│ → Random Forest 300 cây         ├──►• Mức tin cậy
                  │ → Ngưỡng quyết định             │   • Bộ chỉ số đánh giá
                  └─────────────────────────────────┘   • 12 đồ thị phân tích
```

## 2.4. Tiêu chí thành công — đặt ra TRƯỚC khi chạy thí nghiệm

Đây là điểm phương pháp luận quan trọng nhất của đồ án. Sáu tiêu chí dưới đây được
ghi vào `README.md` **trước** khi thí nghiệm đầu tiên được chạy:

| # | Tiêu chí | Ngưỡng |
| --- | --- | --- |
| 1 | Accuracy trên validate | `> 0.55` **và** cao hơn **mọi** mô hình đối chứng |
| 2 | ROC-AUC trên validate | `> 0.55` — tách rõ khỏi mức ngẫu nhiên 0.50 |
| 3 | Chênh lệch Accuracy train ↔ validate | `< 0.10` (dấu hiệu quá khớp) |
| 4 | Độ lệch chuẩn Accuracy qua các vòng walk-forward | `< 0.05` |
| 5 | F1-score của **cả hai** lớp | `≥ 0.40` (mô hình không "bỏ rơi" một lớp) |
| 6 | Suy giảm từ validate sang test | `< 0.05` |

Lý do phải làm như vậy: **nếu đặt tiêu chí sau khi đã thấy kết quả thì tiêu chí nào
cũng đạt** — chỉ cần chọn ngưỡng vừa khít con số mình có. Cách đó không chứng minh
được điều gì. [§5.11](#511-đối-chiếu-sáu-tiêu-chí-chấp-nhận) quay lại đúng sáu dòng
này và đối chiếu từng dòng, kể cả dòng không đạt.

## 2.5. Phạm vi và giới hạn tuyên bố ngay từ đầu

Đồ án **không** đặt mục tiêu xây dựng một hệ thống giao dịch sinh lời. Cụ thể:

- **Không tính chi phí giao dịch, trượt giá, thanh khoản.** Accuracy 0.52 rất có
  thể không sinh lời sau phí.
- **Không dùng thông tin ngoài phân tích kỹ thuật** — tin tức, báo cáo tài chính,
  dữ liệu vĩ mô, dòng tiền khối ngoại đều nằm ngoài tập đặc trưng.
- **Không khuyến nghị đầu tư.** Đây là sản phẩm học thuật minh hoạ quy trình học
  máy trên dữ liệu chuỗi thời gian.

Kỳ vọng hợp lý được xác lập trước: với dữ liệu tài chính, **Accuracy 55–60% đã là
kết quả tốt**. Nếu một mô hình cho Accuracy > 70%, khả năng cao nhất là **có rò rỉ
dữ liệu** — và [§5.3](#53-thí-nghiệm-rò-rỉ-có-kiểm-soát--vì-sao-có-báo-cáo-đạt-90)
chứng minh điều đó bằng thực nghiệm.

---

# 3. DỮ LIỆU

## 3.1. Nguồn và quy mô

| Hạng mục | Giá trị |
| --- | --- |
| Sàn giao dịch | HOSE (Sở GDCK TP. Hồ Chí Minh) |
| Mã cổ phiếu | **VNM** (Vinamilk), **VCB** (Vietcombank), **MSN** (Masan), **MWG** (Thế Giới Di Động) |
| Khoảng thời gian | 2016-08-22 → 2026-08-21 (~10 năm) |
| Số phiên thô (VNM) | 2 587 |
| Số mẫu sau dựng đặc trưng | 2 456 |
| Định dạng | `.csv` / `.xlsx` — cột `Date, Open, High, Low, Close, Volume` |

Bốn mã được chọn thuộc bốn ngành khác nhau (tiêu dùng, ngân hàng, đa ngành, bán lẻ)
để tránh việc cả bốn cùng chịu chung một cú sốc ngành.

## 3.2. Vì sao chọn đúng bốn mã này — một quyết định có kiểm soát

Việc chọn mã **không** ngẫu nhiên và cũng **không** dựa trên mã nào cho kết quả đẹp.
Quy trình: chấm điểm **mức lệch phân phối** của giai đoạn gần đây so với giai đoạn
trước đó cho **12 mã**, rồi giữ lại bốn mã ổn định nhất.

| Mã | Điểm lệch | Quyết định |
| --- | --- | --- |
| VNM, VCB, MSN, MWG | thấp nhất trong 12 mã | ✅ Giữ lại |
| FPT | 0.341 | ❌ Loại |
| VIC | 1.119 | ❌ Loại |

Lý do: nếu để một mã có **biến cố cấu trúc lớn** (chia tách, sáp nhập, khủng hoảng
riêng) lọt vào tập thí nghiệm thì khi mô hình cho kết quả kém, ta **không tách được**
nguyên nhân "mô hình dở" khỏi nguyên nhân "thị trường đổi chế độ". Loại chúng ra là
để giữ cho phép đo có ý nghĩa.

## 3.3. Chất lượng dữ liệu

| Kiểm tra | Kết quả | Đánh giá |
| --- | --- | --- |
| Ô thiếu giá trị | 0 trên mọi cột | Không cần điền tiến |
| Dòng bị loại khi làm sạch | 0 | Chuỗi liền mạch |
| Khoảng cách lớn nhất giữa hai phiên | 8 ngày | Trong mức nghỉ lễ thông thường |
| Biến động > 40% trong một phiên | 0 | Không có dấu hiệu lỗi dữ liệu |
| Quan sát đứng yên (`h = 5`) | 81 (3.14%) | Đã loại khỏi tập huấn luyện |

## 3.4. Xử lý quan sát đứng yên — một quyết định tiền xử lý có ảnh hưởng quyết định

Ở tầm nhìn **1 phiên**, VNM có **355 phiên đứng yên (13.7%)** do giá được làm tròn
theo bước giá của sàn. Cách xử lý nhóm này **thay đổi hẳn bài toán**:

| Cách xử lý | Phân bố lớp (giảm/tăng) | Mốc majority | Hệ quả |
| --- | --- | --- | --- |
| Dồn hết vào lớp "giảm" | 1 501 / 1 036 = **59.2 / 40.8** | 0.5916 | Mất cân bằng **giả**; mô hình học cách luôn đoán "giảm", F1 lớp "tăng" chỉ 0.1878 |
| **Loại khỏi tập huấn luyện** | 1 154 / 1 036 = **52.7 / 47.3** | 0.5269 | Bài toán cân bằng, mô hình buộc phải học tín hiệu thật |

Đồ án chọn cách thứ hai (`flat_label = null` trong `config/base.json`).

> **Nhận định:** đây là ví dụ điển hình cho việc một lựa chọn tiền xử lý tưởng như
> vụn vặt lại quyết định toàn bộ kết quả. Chênh lệch mốc majority 0.59 ↔ 0.53 lớn
> hơn tác động của **mọi** phép tinh chỉnh siêu tham số cộng lại.

## 3.5. Số mẫu qua từng bước

```text
2 587 phiên thô
  → 2 587 sau làm sạch            (không mất dòng nào)
  → 2 456 sau dựng đặc trưng      (mất 131 dòng do: 50 phiên khởi động cửa sổ
                                   SMA-50 · 5 dòng cuối không có nhãn ở h = 5 ·
                                   81 quan sát đứng yên — ba nhóm có phần
                                   chồng lấn nhau)
  → 1 719 / 368 / 359             (train / validate / test, cách nhau 2 gap × 5 phiên)
```

| Tập | Số mẫu | Tỷ lệ | Khoảng thời gian |
| --- | --- | --- | --- |
| Train | 1 719 | 70.3% | 2016-10-31 → 2023-09-06 |
| Validate | 368 | 15.0% | 2023-09-14 → 2025-03-12 |
| Test | 359 | 14.7% | 2025-03-20 → 2026-08-14 |

Kiểm chứng: `1 719 + 5 + 368 + 5 + 359 = 2 456` — khớp chính xác.

---

# 4. PHƯƠNG PHÁP HỌC MÁY

## 4.1. Quy trình tám bước

```text
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ 1. Nạp DL  │ → │ 2. Tiền xử │ → │ 3. Đặc     │ → │ 4. Gán     │
│  CSV/XLSX  │   │  lý t.gian │   │  trưng KT  │   │  nhãn      │
└────────────┘   └────────────┘   └────────────┘   └─────┬──────┘
                                                         │
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌─────▼──────┐
│ 8. Báo cáo │ ← │ 7. Đánh    │ ← │ 6. Tinh    │ ← │ 5. Tách    │
│  & lưu KQ  │   │  giá       │   │  chỉnh HP  │   │  tập t.gian│
└────────────┘   └────────────┘   └─────┬──────┘   └────────────┘
                                        │ ↑
                                  ┌─────▼──────┐
                                  │ Huấn luyện │
                                  │ Random For.│
                                  └────────────┘
```

Mỗi bước là **một module riêng có thể kiểm thử độc lập**:

| Bước | Module | Nội dung |
| --- | --- | --- |
| 1 | `utilities/dataLoader.py` | Đọc `.csv`/`.xlsx`, làm sạch **theo dòng** để các cột luôn khớp chỉ số |
| 2 | `pipeline/timePreprocess.py` | Sắp xếp tăng dần, khử ngày trùng, điền tiến ≤ 2 phiên, phát hiện biến động bất thường |
| 3 | `pipeline/technicalIndicators.py` + `featureBuilder.py` | 20 chỉ báo, thi hành **bản đặc tả khai báo** từ file cấu hình |
| 4 | `pipeline/labeling.py` | Nhãn xu hướng, mục tiêu hồi quy, loại quan sát đứng yên |
| 5 | `pipeline/splitter.py` | Tách theo thứ tự thời gian có gap, sinh vòng walk-forward |
| 6 | `libraries/randomForest.py` | Bagging + random feature + OOB + importance |
| 7 | — | Dò siêu tham số bằng `TimeSeriesSplit`, **không** dùng `KFold` ngẫu nhiên |
| 8 | `utilities/metrics*.py` + `libraries/rfPlot.py` | Bộ chỉ số đầy đủ + 14 hàm vẽ |

## 4.2. Lựa chọn tầm nhìn dự báo

Đề bài không quy định tầm nhìn, nên đồ án **đo cả hai lựa chọn** với cùng bộ đặc
trưng và cùng quy trình:

| Mã | Tầm nhìn | Mốc majority (validate) | Accuracy validate | ROC-AUC validate | Chênh train↔validate |
| --- | --- | --- | --- | --- | --- |
| VNM | 1 phiên | 0.5579 | 0.5793 | 0.5838 | 0.1298 |
| **VNM** | **5 phiên** | **0.5788** | **0.6223** | **0.6444** | **0.0374** |
| FPT | 1 phiên | 0.5435 | 0.5766 | 0.6052 | 0.0828 |
| FPT | 5 phiên | 0.5550 | 0.5255 | 0.4987 | 0.1542 |

*(cấu hình đối chiếu: `max_depth = 5`, `min_samples_leaf = 40`, 150 cây)*

Hai điều rút ra:

1. **Với VNM, tầm nhìn 5 phiên cho tín hiệu rõ hơn hẳn.** Ở tầm nhìn 1 phiên, phần
   lớn biến động là nhiễu vi cấu trúc và tỷ lệ phiên đi ngang rất cao; kéo dài tầm
   nhìn giúp phần biến động có cấu trúc nổi lên trên nền nhiễu. 5 phiên = một tuần
   giao dịch, đủ để tín hiệu kỹ thuật thể hiện.
2. **Tín hiệu không phổ quát giữa các mã.** Với FPT, tầm nhìn 5 phiên cho ROC-AUC
   0.4987 — không phân biệt được gì so với đoán bừa. Đây là phát hiện đáng ghi
   nhận: **một quy trình chạy tốt trên mã này hoàn toàn có thể vô dụng trên mã
   khác**, và không được suy rộng kết quả từ một mã ra cả thị trường.

## 4.3. Bộ đặc trưng — 27 chỉ báo phân tích kỹ thuật

Toàn bộ đặc trưng được khai báo dưới dạng **bản đặc tả JSON**, không phải mã Python.
Một mục có dạng:

```json
{
  "name":      "close_over_sma20",
  "indicator": "ratio_to_moving_average",
  "inputs":    ["close"],
  "params":    {"window": 20},
  "keep":      true
}
```

`inputs` tra lần lượt trong các dãy gốc rồi tới các dãy đã dựng ở những mục **trước
đó**, nên bản đặc tả có thể **xếp tầng** (ví dụ `atr_raw` → `atr_over_close`). Cờ
`keep: false` cho phép dựng một dãy trung gian mà không đưa nó vào tập đặc trưng
cuối.

| Nhóm | Đặc trưng | Số lượng |
| --- | --- | --- |
| **Xu hướng** | `close_over_sma5/10/20/50`, `close_over_ema12`, `sma5_above_sma20`, `macd_histogram`, `macd_over_signal` | 8 |
| **Động lượng** | `rsi_7`, `rsi_14`, `roc_5`, `roc_10`, `stochastic_k`, `stochastic_d`, `return_lag_1..5` | 11 |
| **Biến động** | `bollinger_b`, `bollinger_width`, `atr_over_close`, `relative_range`, `volatility_10`, `volatility_20` | 6 |
| **Khối lượng** | `volume_ratio_20`, `obv_return` | 2 |
| | **Tổng** | **27** |

Ba nguyên tắc thiết kế đặc trưng:

1. **Ưu tiên đặc trưng dạng tỷ lệ** (`Close/SMA20`, `%B`, `return`) thay vì giá
   tuyệt đối. Random Forest **không ngoại suy được**: huấn luyện trên vùng giá
   20–30 thì mô hình không bao giờ dự đoán ra giá 50. Đặc trưng dạng tỷ lệ giữ
   nguyên ý nghĩa qua mọi mức giá.
2. **Mọi chỉ báo phải nhân quả** — giá trị tại `t` chỉ được tính từ dữ liệu `≤ t`.
   Tính chất này được kiểm thử tự động, xem [§5.1](#51-kiểm-chứng-cài-đặt-thuật-toán).
3. **Giữ nguyên cả 27, không cắt bớt theo tầm quan trọng.** Việc chọn lọc đặc trưng
   dựa trên tập validate lại chính là một dạng rò rỉ khác.

## 4.4. Bốn lớp chống rò rỉ dữ liệu

Rò rỉ dữ liệu (data leakage) là khi mô hình vô tình nhìn thấy tương lai trong lúc
huấn luyện. Trên dữ liệu chuỗi thời gian, đây là **lỗi phổ biến nhất của cả lĩnh
vực**. Hệ thống chặn bằng bốn lớp:

| # | Nguyên tắc | Cài đặt |
| --- | --- | --- |
| 1 | **Chỉ báo nhân quả** | Mọi cửa sổ trượt chỉ đọc quá khứ. Có kiểm thử cắt chuỗi để xác minh |
| 2 | **Nhãn dịch về sau** | Nhãn của `t` lấy từ `Close(t+5)`; **không** dịch đặc trưng về trước |
| 3 | **Tách theo thời gian** | `shuffle = False` tuyệt đối; train luôn nằm trước validate |
| 4 | **`gap = horizon = 5`** | Chặn mẫu cuối train chạm vào mẫu đầu validate |

Lớp thứ tư tinh vi nhất và đáng giải thích kỹ: nhãn nhìn về tương lai 5 phiên. Nếu
tập train kết thúc đúng ngày tập validate bắt đầu, thì **5 mẫu cuối cùng của tập
train mang nhãn được tính từ giá nằm trong tập validate**. Chỉ 5 mẫu, nhưng vẫn là
rò rỉ. Vì vậy hệ thống bỏ đúng 5 phiên ở mỗi ranh giới.

> **Bốn nguyên tắc này làm kết quả xấu đi.** Đó chính là lý do tin rằng chúng đúng.

## 4.5. Mô hình

### 4.5.1. Random Forest — cài đặt thuần Python

Ba trụ cột của thuật toán được cài trong `src/libraries/`:

| Thành phần | Module | Nội dung |
| --- | --- | --- |
| Cây quyết định | `decisionTree.py` | CART đệ quy, tìm ngưỡng chia tối ưu, điều kiện dừng |
| Tiêu chí phân tách | `rfMath.py` | Gini, Entropy, Information Gain; variance reduction cho hồi quy |
| Bagging | `randomForest.py` | Bootstrap `n` mẫu **có hoàn lại** cho mỗi cây |
| Ngẫu nhiên hoá đặc trưng | `randomForest.py` | Xét ngẫu nhiên `m = √p` đặc trưng tại **mỗi nút** |
| Tổng hợp dự đoán | `randomForest.py` | **Bỏ phiếu mềm** theo xác suất (không phải bỏ phiếu cứng) |
| OOB & Importance | `randomForest.py` | Lỗi out-of-bag, MDI, permutation importance |

Tầng `src/libraries/` **không phụ thuộc bất kỳ thư viện ngoài nào** — chạy được với
Python chuẩn, và **không chứa một dòng nào nhắc tới "giá", "cổ phiếu" hay tên mã**.
Nó chỉ biết ma trận và nhãn.

### 4.5.2. Các mô hình đối chứng

Với dữ liệu tài chính, mọi kết quả **bắt buộc** phải được so với các mốc tầm thường:

| Đối chứng | Cách dự đoán | Vai trò |
| --- | --- | --- |
| **Majority class** | Luôn dự đoán lớp chiếm đa số | Mốc sàn tuyệt đối |
| **Naive persistence** | Xu hướng hôm nay lặp lại ngày mai | Mốc sàn theo chuỗi thời gian |
| **Alternating** | Đổi chiều luân phiên | Mốc kiểm tra ROC-AUC ≈ 0.5 |
| **Logistic Regression** | Mô hình tuyến tính, **cùng 27 đặc trưng** | Kiểm tra RF có thực sự khai thác quan hệ phi tuyến |
| **Mean-of-train** (nhánh B) | Luôn dự đoán trung bình tập train | Mốc để `R²` có ý nghĩa |
| **Gradient Boosting** | Ensemble giảm **độ chệch** thay vì phương sai | So sánh hai họ ensemble |

Hồi quy tuyến tính và logistic cũng được **tự cài đặt thuần Python** trong
`libraries/linearModel.py`, có kiểm thử riêng đối chiếu với `scikit-learn`.

## 4.6. Siêu tham số cuối cùng

| Tham số | Nhánh A (phân loại) | Nhánh B (hồi quy) | Lý do chọn |
| --- | --- | --- | --- |
| `n_estimators` | 300 | 300 | Đường cong OOB đã đi ngang trước mốc này |
| `criterion` | `gini` | `variance` | Mặc định chuẩn cho từng loại bài toán |
| `max_depth` | **5** | 6 | Ràng buộc mạnh — xem [§5.4](#54-vì-sao-phải-ràng-buộc-mô-hình) |
| `min_samples_leaf` | **40** | 30 | Lá nhỏ hơn là học thuộc nhiễu |
| `max_features` | `sqrt` → 5/27 | `third` → 9/27 | Giá trị khuyến nghị cho phân loại / hồi quy |
| `bootstrap` | `True` | `True` | Bắt buộc để có ước lượng OOB |
| `random_state` | 42 | 42 | Cố định để tái lập |

> **Ghi chú về tính trung thực:** siêu tham số được dò **chỉ trên VNM**. Ba mã còn
> lại (VCB, MSN, MWG) **không được chạm vào tham số một lần nào** — chúng là kiểm
> chứng sạch. Điều này được ghi thẳng vào `config/vnm.json` để người đọc mã nguồn
> biết kết quả của VNM lạc quan hơn ba mã kia.

---

# 5. KẾT QUẢ THÍ NGHIỆM

Toàn bộ số liệu trong mục này sinh ra từ `src/mainClassification.py`,
`src/mainRegression.py` và `docs/presentation/makeEvidence.py` với
`random_state = 42`, tái lập được nguyên vẹn.

## 5.1. Kiểm chứng cài đặt thuật toán

Câu hỏi đầu tiên phải trả lời: **tự cài đặt thì lấy gì đảm bảo đúng?** Ba cách kiểm
chứng độc lập:

### (a) Tính chất toán học có đáp án biết trước

Khi lấy mẫu bootstrap `n` lần từ `n` mẫu, tỷ lệ mẫu **không** được chọn hội tụ về
`1/e`:

| Đại lượng | Giá trị |
| --- | --- |
| Tỷ lệ OOB đo được trong rừng | **0.3676** |
| Công thức `(1 − 1/n)ⁿ` | 0.3676 |
| Giới hạn `1/e` | 0.3679 |
| Phép đo độc lập trong `makeEvidence.py` | 0.36734 (lý thuyết 0.36788) |

Khớp đến bốn chữ số thập phân. Đây là kiểm chứng đáng tin nhất vì **không thể ăn
may** — một cài đặt bootstrap sai sẽ không ngẫu nhiên rơi trúng `1/e`.

### (b) Đối chiếu với `scikit-learn`

Cùng dữ liệu, cùng siêu tham số, cùng hạt giống:

| Cài đặt | Accuracy validate | ROC-AUC validate | Accuracy test | Lỗi OOB |
| --- | --- | --- | --- | --- |
| **Thuần Python (đồ án)** | 0.6087 | 0.6476 | 0.5237 | 0.3973 |
| `scikit-learn` | 0.6141 | 0.6446 | 0.5181 | 0.4043 |
| **Chênh lệch** | **0.0054** | **0.0030** | **0.0056** | **0.0070** |

Chênh lệch **dưới 1% ở cả bốn chỉ số**. Phần khác biệt còn lại đến từ chi tiết sinh
số ngẫu nhiên và cách chọn ngưỡng chia — hai cài đặt độc lập không thể trùng khít
tuyệt đối.

Phép kiểm chứng chặt hơn — hai cài đặt có **đồng ý về đặc trưng nào quan trọng**
hay không:

| Đặc trưng | Hạng (đồ án) | Hạng (sklearn) | Lệch |
| --- | --- | --- | --- |
| `volatility_10` | 1 | 1 | 0 |
| `atr_over_close` | 2 | 2 | 0 |
| `close_over_sma50` | 3 | 3 | 0 |
| `volatility_20` | 4 | 4 | 0 |
| `rsi_14` | 5 | 6 | 1 |
| `roc_10` | 8 | 5 | 3 |

**Bốn đặc trưng hàng đầu trùng khớp tuyệt đối về thứ hạng.**

> **Lưu ý về vai trò của `scikit-learn`:** thư viện này được dùng để **kiểm chứng**,
> không dùng để chạy. Toàn bộ kết quả báo cáo đến từ mã tự viết.

### (c) Bộ kiểm thử tự động

```powershell
python -m pytest tests/ -q
# → 222 passed
```

**222 test đã được chạy lại và ĐẠT toàn bộ** khi lập báo cáo này (28.1 giây). Bộ
kiểm thử tập trung vào bốn tính chất mà nếu sai thì mọi con số đánh giá phía sau đều
vô nghĩa:

| Nhóm | Kiểm tra điều gì |
| --- | --- |
| **Tính nhân quả** | Cắt chuỗi tại vị trí 1 000, tính lại chỉ báo trên đoạn đầu; giá trị trước điểm cắt phải **không đổi**. Cả 5 chỉ báo được kiểm tra đều ĐẠT |
| **Đúng công thức** | Đối chiếu Gini, Entropy, Precision/Recall/F1, ROC-AUC với giá trị **tính tay** trên ví dụ nhỏ |
| **Đúng lý thuyết** | Tỷ lệ OOB hội tụ `1/e`; `R²` của dự đoán bằng trung bình đúng bằng 0; cây hồi quy **không** ngoại suy được |
| **Tái lập được** | Cùng `random_state` cho cùng kết quả; đổi hạt giống cho rừng khác |

## 5.2. Phép thử xáo nhãn — kiểm tra rò rỉ

Đây là thí nghiệm quan trọng nhất về mặt phương pháp. Cách làm: **xáo ngẫu nhiên
toàn bộ nhãn huấn luyện** — phá sạch mọi mối liên hệ giữa đặc trưng và nhãn — rồi
huấn luyện lại y hệt.

| | Nhãn thật | Nhãn xáo lần 1 | Nhãn xáo lần 2 | Nhãn xáo lần 3 |
| --- | --- | --- | --- | --- |
| Accuracy train | 0.6597 | 0.6719 | 0.6632 | 0.6812 |
| **ROC-AUC validate** | **0.6418** | **0.4134** | **0.4535** | **0.5445** |

Thí nghiệm này trả lời **hai** câu hỏi:

**① Quy trình có rò rỉ không?** Nếu có, mô hình vẫn sẽ tìm được đường tắt và AUC
validate vẫn cao dù nhãn đã bị phá. Kết quả: AUC sụp xuống 0.41–0.54, tức **quanh
mức ngẫu nhiên, có lần còn dưới**. Không có đường tắt nào tồn tại trong dữ liệu.
**Quy trình sạch.**

**② Accuracy train có ý nghĩa gì không?** Nhãn thật cho 0.6597; nhãn **ngẫu nhiên
hoàn toàn** cho 0.6719 / 0.6632 / 0.6812 — **cao hơn**. Nghĩa là Accuracy train 66%
**không chứa một chút thông tin nào** về chất lượng mô hình; nó chỉ đo khả năng học
thuộc. Vì vậy trong toàn bộ đồ án, Accuracy train **không** được dùng để kết luận
bất cứ điều gì.

## 5.3. Thí nghiệm rò rỉ có kiểm soát — vì sao có báo cáo đạt 90%

Thí nghiệm này trả lời câu hỏi *"sao accuracy thấp thế, nhiều báo cáo khác đạt
90%?"* bằng số liệu thay vì lập luận. **Cùng một bộ dữ liệu, cùng một thuật toán,
cùng một đoạn mã** — thứ duy nhất thay đổi là **cách tách tập**:

| # | Cách làm | Accuracy | Đánh giá |
| --- | --- | --- | --- |
| 1 | Tách **theo thời gian** (đồ án dùng) | **0.5292** | ✅ Trung thực |
| 2 | Xáo trộn ngẫu nhiên rồi tách 85/15 | 0.5718 | ⚠️ **+4.3 điểm miễn phí** |
| 3 | Xáo trộn + để cây mọc tự do | 0.6612 | ⚠️ **+13.2 điểm** |
| 4 | Báo cáo trên **chính tập huấn luyện** | **1.0000** | ❌ 100%, vô giá trị |

Giải thích từng dòng:

- **Dòng 2** là hành vi mặc định của `train_test_split` — khi xáo trộn, mẫu của
  ngày mai và ngày hôm nay cùng nằm trong tập huấn luyện, mà hai mẫu liền kề trong
  chuỗi thời gian thì gần như giống hệt nhau. **Mô hình không dự đoán — nó nhận ra
  hàng xóm.**
- **Dòng 3** cộng thêm việc bỏ ràng buộc độ sâu, cho cây ghi nhớ từng mẫu.
- **Dòng 4** là báo cáo hiệu năng trên chính dữ liệu đã huấn luyện.

> **Hệ quả cho việc so sánh:** con số 0.5292 của đồ án và con số 0.90 của một báo
> cáo khác **không so sánh được với nhau**, vì chúng đo hai thứ khác nhau. Đồ án
> hoàn toàn có thể tạo ra con số 0.66 hoặc 1.00 trong vòng năm phút — và chọn không
> làm vậy.

## 5.4. Vì sao phải ràng buộc mô hình

Bảng dưới đo trực tiếp hiện tượng quá khớp — cùng dữ liệu, cùng 100 cây, chỉ đổi
ràng buộc:

| Ràng buộc | Accuracy train | Accuracy validate | Chênh lệch | ROC-AUC validate |
| --- | --- | --- | --- | --- |
| `depth = ∞, leaf = 1` | **1.0000** | 0.5707 | **0.4293** | 0.6066 |
| `depth = 12, leaf = 10` | 0.9436 | 0.6033 | 0.3403 | 0.6354 |
| `depth = 8, leaf = 20` | 0.7807 | 0.6196 | 0.1611 | 0.6360 |
| **`depth = 5, leaf = 40`** | **0.6597** | **0.6250** | **0.0347** | **0.6418** |

Cây mọc tự do ghi nhớ **hoàn hảo** tập huấn luyện nhưng cho kết quả validate **thấp
nhất bảng ở cả hai chỉ số**. Toàn bộ phần chênh lệch 0.4293 là học thuộc nhiễu.

Càng siết ràng buộc, khoảng cách càng thu hẹp **và** kết quả validate càng tốt lên.
Quan hệ ngược chiều này là bằng chứng rõ ràng nhất cho thấy **dữ liệu tài chính đòi
hỏi mô hình đơn giản**.

## 5.5. Nhánh A — kết quả phân loại trên VNM

### 5.5.1. Kết quả trên ba tập

| Tập | Accuracy | ROC-AUC | F1 (tăng) | F1 (giảm) | Balanced Acc. | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 0.6597 | 0.7663 | 0.5524 | 0.7255 | 0.6480 | +0.3260 |
| **Validate** | **0.6087** | **0.6476** | **0.4098** | **0.7073** | **0.5697** | **+0.1608** |
| Test | 0.5237 | 0.5313 | 0.4706 | 0.5671 | 0.5215 | +0.0436 |

Báo cáo chi tiết trên tập validate:

```text
Lớp            Precision      Recall    F1-score   Support
----------------------------------------------------------
0 (giảm)          0.6237      0.8169      0.7073       213
1 (tăng)          0.5618      0.3226      0.4098       155
----------------------------------------------------------
Macro avg         0.5927      0.5697      0.5586
Weighted avg      0.5976      0.6087      0.5820
Accuracy          0.6087
```

**Diễn giải hành vi mô hình:** `Precision` lớp "tăng" đạt 0.5618 trong khi `Recall`
chỉ 0.3226. Mô hình **thận trọng** — chỉ báo "tăng" khi khá chắc, chấp nhận bỏ sót
nhiều cơ hội. Với ứng dụng thực tế, đây thường là hành vi mong muốn hơn chiều ngược
lại, vì mỗi tín hiệu sai đều phải trả giá bằng chi phí giao dịch.

### 5.5.2. Ổn định qua kiểm định tiến dần (walk-forward)

| Vòng | Số mẫu train | Số mẫu validate | Accuracy |
| --- | --- | --- | --- |
| 1 | 1 043 | 207 | 0.5797 |
| 2 | 1 250 | 207 | 0.6039 |
| 3 | 1 457 | 207 | 0.5749 |
| 4 | 1 664 | 207 | 0.5942 |
| 5 | 1 871 | 211 | 0.5545 |
| | | **Trung bình** | **0.5814 ± 0.0170** |

Độ lệch chuẩn **1.70%** — mô hình ổn định giữa các giai đoạn, không phải ăn may ở
một đoạn dữ liệu cụ thể.

### 5.5.3. So với các mốc đối chứng

| Tập | Mô hình | majority | persistence | alternating | Kết luận |
| --- | --- | --- | --- | --- | --- |
| Validate | **0.6087** | 0.5788 | 0.5326 | 0.5136 | **Vượt cả ba** (+3.0 điểm so với mốc cao nhất) |
| Test | **0.5237** | 0.5125 | 0.4708 | 0.4708 | Vượt cả ba, nhưng chỉ **+1.1 điểm** |

### 5.5.4. Ngưỡng quyết định không chuyển giao

Ngưỡng được dò trên tập validate theo `balanced accuracy`, sau đó áp **nguyên vẹn**
lên tập test:

| Ngưỡng | Nguồn | Accuracy test | Balanced Acc. test |
| --- | --- | --- | --- |
| 0.50 | mặc định | 0.5237 | 0.5215 |
| 0.45 | tối ưu trên validate (đạt 0.6237 ở đó) | 0.5153 | 0.5175 |

Ngưỡng tối ưu trên validate **làm kết quả test xấu đi**. Đây là bằng chứng trực
tiếp cho thấy quan hệ giữa đặc trưng và mục tiêu **đã đổi** giữa hai giai đoạn.

## 5.6. Bốn mã, một bộ siêu tham số — kiểm định thống kê McNemar

Bốn mô hình riêng cho bốn mã, nhưng dùng **một bộ siêu tham số duy nhất** dò **chỉ
trên VNM**:

| Mã | Valid acc | Valid đoán bừa | p (McNemar) | Test acc | Test đoán bừa | p (McNemar) |
| --- | --- | --- | --- | --- | --- | --- |
| **VNM** *(mã dò tham số)* | **0.6087** | 0.5788 | 0.289 | 0.5237 | 0.5125 | 0.805 |
| VCB | 0.5359 | 0.5000 | 0.089 | 0.4840 | 0.4429 | 0.063 |
| MSN | 0.5669 | 0.5261 | 0.208 | 0.5219 | 0.4873 | 0.242 |
| MWG | 0.4955 | 0.5402 | **0.042** ⚠️ | 0.5318 | 0.5295 | 1.000 |

**Kiểm định McNemar** chỉ nhìn những mẫu mà hai mô hình **bất đồng** — phần hai mô
hình giống nhau không phân biệt được chúng. Nếu chênh lệch chỉ do may rủi thì số
lần A-đúng-B-sai sẽ xấp xỉ số lần ngược lại.

Hai kết luận, cả hai đều không đẹp và cả hai đều được giữ nguyên trong báo cáo:

1. **Không mã nào có ưu thế dương đạt ý nghĩa thống kê ở mức 5%.** VNM cho
   `p = 0.289` — không thể bác bỏ giả thuyết "chênh lệch này là ngẫu nhiên".
2. **MWG trên validate cho `p = 0.042`, tức CÓ ý nghĩa thống kê — nhưng theo chiều
   ngược lại.** Mô hình tệ hơn đoán bừa một cách có ý nghĩa.

Kết quả ba mã không dò tham số đều thấp hơn VNM — **đúng như dự đoán nếu tồn tại
thiên lệch chọn mô hình**. Giữ VNM lại chính là để cho thấy **độ lớn** của thiên
lệch đó.

## 5.7. Nhánh B — dự đoán độ lớn biến động

### 5.7.1. Kết quả trên ba tập (VNM)

| Tập | RMSE | MAE | R² | MAPE (%) | Directional Accuracy |
| --- | --- | --- | --- | --- | --- |
| Train | 0.029021 | 0.021137 | +0.1836 | 103.4 | 0.6797 |
| **Validate** | **0.022281** | **0.016770** | **+0.0716** | 101.0 | **0.6137** |
| Test | 0.038519 | 0.026981 | −0.0257 | 118.7 | 0.5192 |

> **Cách đọc MAPE cho đúng:** giá trị trên 100% **không** có nghĩa mô hình sai hoàn
> toàn. Mục tiêu là tỷ suất biến động dao động quanh 0, nên mẫu số `|y|` thường rất
> nhỏ và làm phần trăm sai số phóng đại. Với loại mục tiêu này, `RMSE` và `R²` mới
> là thước đo đáng tin — MAPE được giữ lại trong bảng **chỉ để cho thấy vì sao
> không nên dùng nó ở đây**.

### 5.7.2. `R²` trên tập validate — bốn mã, ba mô hình

| Mã | Random Forest | Hồi quy tuyến tính | Đoán bằng trung bình |
| --- | --- | --- | --- |
| VNM | **+0.0716** | −0.0180 | −0.0148 |
| VCB | −0.0420 | −0.0800 | **−0.0154** |
| MSN | −0.0391 | −0.0527 | **−0.0174** |
| MWG | −0.0101 | −0.0198 | **−0.0002** |

*VNM: `SST = 0.20266` · `SSE = 0.18815` · `RMSE = 0.02228` · `MAE = 0.01677`*

**Ý nghĩa của `R²` âm.** Vì `R² = 1 − SSE/SST`, giá trị âm nghĩa là `SSE > SST` —
sai số của mô hình **lớn hơn** sai số của việc đoán bừa bằng trung bình tập huấn
luyện. Cột "đoán bằng trung bình" chính là mô hình **không học gì cả**, và nó
**thắng ở ba trên bốn mã**.

Chỉ VNM cho `R²` dương, và cũng chỉ 0.0716 — mô hình giải thích được **7.2% biến
động**, còn **92.8% là nhiễu**.

### 5.7.3. Giới hạn không ngoại suy — quan sát trực tiếp

| Đại lượng (VNM) | Khoảng giá trị | Biên độ |
| --- | --- | --- |
| Mục tiêu thực tế trong tập train | [−0.1371, +0.1677] | 0.3048 |
| Mục tiêu thực tế trong tập test | [−0.1447, +0.1596] | 0.3043 |
| **Dự đoán trên tập test** | **[−0.0112, +0.0336]** | **0.0448** |
| Biên độ dự đoán trên validate | — | 0.0389 (thực tế 0.1580) |

Dải dự đoán **hẹp hơn khoảng 4–7 lần** dải thực tế. Nguyên nhân mang tính **cấu
trúc**: rừng lấy trung bình dự đoán của 300 cây, mà mỗi cây lại lấy trung bình các
giá trị trong lá — **hai lớp lấy trung bình chồng lên nhau san phẳng mọi giá trị
cực đoan**.

Hệ quả thực tế: **mô hình không bao giờ dự báo được các phiên biến động mạnh** —
đúng những phiên đáng quan tâm nhất. Mô hình đã học được điều đúng đắn nhất mà nó có
thể học: *"đừng đoán mạnh"*.

### 5.7.4. Bắc cầu sang nhánh A

`Directional Accuracy` là chỉ số chung của hai nhánh:

| Tập | Nhánh A (học thẳng chiều) | Nhánh B (suy từ dấu tỷ suất) |
| --- | --- | --- |
| Validate | 0.6087 | 0.6137 |
| Test | 0.5237 | 0.5192 |

Hai nhánh cho kết quả **gần như trùng nhau**. Nghĩa là **bắt mô hình học thêm độ
lớn của biến động không giúp nó đoán hướng tốt hơn** — phần thông tin dùng được nằm
ở **dấu**, còn độ lớn hầu như chỉ là nhiễu.

> **Đây là căn cứ để đóng khung đề tài thành bài toán phân loại.** Dự đoán độ lớn
> biến động là bài toán mà dữ liệu này không giải được; dự đoán hướng thì còn một
> chút tín hiệu — và "một chút" đó là toàn bộ những gì đồ án báo cáo.

## 5.8. So sánh các họ mô hình

Tất cả chạy trên **cùng bộ đặc trưng và cùng cách tách tập** (VNM):

| Mô hình | Accuracy validate | AUC validate | Accuracy test | AUC test |
| --- | --- | --- | --- | --- |
| RF (thuần Python) | 0.6087 | **0.6476** | 0.5237 | 0.5313 |
| RF (`scikit-learn`) | **0.6141** | 0.6446 | 0.5181 | 0.5306 |
| GB (thuần Python) | **0.6141** | 0.6214 | 0.5404 | 0.5556 |
| GB (`scikit-learn`) | 0.5870 | 0.6127 | 0.5599 | 0.5585 |
| **Logistic Regression** | 0.5435 | 0.5081 | **0.5850** | **0.6086** |
| *Mốc majority* | *0.5788* | *0.5000* | *0.5125* | *0.5000* |

Mở rộng ra bốn mã, so trực tiếp Random Forest ↔ Logistic Regression:

| | VNM | VCB | MSN | MWG | Tổng kết |
| --- | --- | --- | --- | --- | --- |
| **Validate** — Rừng | **0.6087** | **0.5359** | **0.5669** | 0.4955 | Rừng thắng **3/4** |
| **Validate** — Hồi quy | 0.5353 | 0.5135 | 0.4739 | **0.5290** | |
| **Test** — Rừng | 0.5237 | 0.4840 | 0.5219 | **0.5318** | Rừng thắng **1/4** |
| **Test** — Hồi quy | **0.5822** | **0.5228** | **0.5473** | 0.4864 | |

**Thứ hạng đảo chiều hoàn toàn giữa hai tập: 3/4 → 1/4.**

Kết luận đúng **không phải** "mô hình tuyến tính tốt hơn" — kết quả tốt của nó trên
test cũng chỉ là một lần quan sát duy nhất. Kết luận đúng là: **ngay cả việc chọn
mô hình nào tốt hơn cũng không ổn định qua các giai đoạn.** Nếu chỉ nhìn tập
validate, ta đã kết luận Random Forest thắng và **tự tin sai**.

> Đây chính là lý do tập test được giữ nguyên và **chỉ mở ra một lần duy nhất ở
> cuối**.

## 5.9. Diễn giải mô hình — đặc trưng nào thực sự được dùng

Top 10 theo MDI, đặt cạnh permutation importance đo trên tập validate. Giá trị
permutation là **mức TĂNG lỗi** khi xáo trộn đặc trưng đó — **giá trị âm nghĩa là
xáo trộn còn làm mô hình tốt lên**, tức đặc trưng không đóng góp gì.

| Hạng | Đặc trưng | MDI | Permutation | Nhóm |
| --- | --- | --- | --- | --- |
| 1 | `volatility_10` | 0.1442 | **+0.04728** | Biến động |
| 2 | `atr_over_close` | 0.1044 | −0.00326 | Biến động |
| 3 | `close_over_sma50` | 0.0850 | +0.00489 | Xu hướng |
| 4 | `volatility_20` | 0.0645 | −0.00543 | Biến động |
| 5 | `rsi_14` | 0.0515 | **+0.01087** | Động lượng |
| 6 | `close_over_sma10` | 0.0482 | −0.00380 | Xu hướng |
| 7 | `bollinger_width` | 0.0448 | −0.00000 | Biến động |
| 8 | `roc_10` | 0.0422 | +0.00272 | Động lượng |
| 9 | `close_over_sma20` | 0.0405 | −0.00000 | Xu hướng |
| 10 | `close_over_ema12` | 0.0368 | −0.00109 | Xu hướng |

Ba nhận xét:

1. **Nhóm biến động chiếm ưu thế.** Bốn trong mười đặc trưng hàng đầu đo **biến
   động** chứ không đo **hướng**, và `volatility_10` đứng đầu ở cả hai cách đo. Mô
   hình học được rằng *mức độ dao động* dự báo được nhiều hơn *chiều dao động* —
   phù hợp với hiểu biết đã có về thị trường tài chính: biến động có tính **cụm**
   (volatility clustering) rõ rệt, còn hướng thì gần với ngẫu nhiên.

2. **Hai cách đo bất đồng mạnh — và permutation mới là cách đáng tin.**
   `atr_over_close` được MDI xếp hạng 2 nhưng permutation **âm**: xáo trộn nó không
   làm mô hình tệ đi. Đây là biểu hiện điển hình của nhóm đặc trưng **tương quan
   cao** — `atr_over_close`, `volatility_10`, `volatility_20` và `bollinger_width`
   cùng đo một thứ, nên phá hỏng một cái thì ba cái còn lại vẫn cung cấp đủ thông
   tin. MDI chia đều công trạng cho cả nhóm và tạo **ảo giác** rằng cả bốn đều quan
   trọng.

3. **Chỉ hai đặc trưng thực sự không thay thế được:** `volatility_10` (+0.047) và
   `rsi_14` (+0.011). Toàn bộ 25 đặc trưng còn lại đóng góp gần bằng 0 hoặc âm. Nói
   cách khác, **bộ 27 đặc trưng có thể rút gọn rất mạnh** mà không mất mát đáng kể.

### Cấu trúc rừng

| Chỉ số | Giá trị |
| --- | --- |
| Số cây | 300 |
| Đặc trưng xét mỗi nút (`m`) | 5 / 27 |
| Độ sâu trung bình | 5.0 (chạm trần `max_depth`) |
| Số lá trung bình | 13.5 |
| Tổng số nút trong | 3 754 |
| Tổng số nút lá | 4 054 |
| **Tổng tham số** | **19 670** |
| Tỷ lệ tham số / mẫu huấn luyện | **11.44** |
| Lỗi OOB | 0.3973 |

## 5.10. Ba thí nghiệm chẩn đoán giới hạn của mô hình

### 5.10.1. Đường cong học — mô hình còn thiếu dữ liệu

Câu hỏi: **60% là do thiếu dữ liệu, hay do dữ liệu vốn không chứa nhiều tín hiệu
hơn?** Huấn luyện lại với 20%, 40%, 60%, 80% và 100% dữ liệu:

| Số mẫu train | 343 | 687 | 1 031 | 1 375 | 1 719 |
| --- | --- | --- | --- | --- | --- |
| ROC-AUC validate | 0.4496 | 0.4985 | 0.5052 | 0.6339 | **0.6418** |

Đường cong **vẫn đang dốc lên ở điểm cuối** — chưa chạm trần. Nghĩa là **thêm dữ
liệu vẫn còn giúp được**, và giới hạn hiện tại một phần đến từ việc chỉ có ~1 700
mẫu huấn luyện. Đây là hướng cải thiện rõ ràng nhất, và nó **không đòi hỏi thuật
toán phức tạp hơn** — chỉ cần nhiều lịch sử hơn.

### 5.10.2. Kích thước mô hình — đang thừa tham số gấp 10 lần

| Cấu hình | Số tham số | AUC validate | Accuracy test |
| --- | --- | --- | --- |
| 300 cây, sâu 5, lá 40 *(đang dùng)* | 19 670 | 0.6476 | 0.5237 |
| 100 cây, sâu 4, lá 60 | 4 135 | **0.6493** | 0.5404 |
| **50 cây, sâu 4, lá 60** | **1 955** | 0.6418 | **0.5460** |
| 30 cây, sâu 3, lá 100 | 735 | 0.6212 | 0.5292 |

Bản nhỏ hơn **mười lần** — chưa tới 2 000 tham số — cho AUC gần như y hệt, và
Accuracy test còn **cao hơn** (0.5460 so với 0.5237). Nói cách khác, **~90% số tham
số hiện tại đang dùng để học thuộc nhiễu chứ không học quy luật**.

> **Vì sao vẫn giữ bản lớn trong báo cáo:** bản 19 670 tham số là bản được dò tham
> số từ đầu theo quy trình. Đổi sang bản nhỏ **sau khi đã nhìn thấy điểm test** thì
> lại chính là một dạng gian lận khác — đúng cái lỗi đồ án phê phán ở
> [§5.3](#53-thí-nghiệm-rò-rỉ-có-kiểm-soát--vì-sao-có-báo-cáo-đạt-90). Phát hiện
> này được ghi nhận như **kiến nghị cho lần làm lại**, không phải như kết quả.

### 5.10.3. Rừng thích nghi — một kết quả âm được đo cẩn thận

Ý tưởng ban đầu: cho rừng **tự cập nhật** khi dữ liệu mới chảy vào — định kỳ trồng
cây mới trên dữ liệu gần đây và loại bớt cây cũ (`libraries/slidingForest.py`).
Bốn phương án được thử, kể cả phương án công bằng nhất (cây mới học trên **đúng
lượng dữ liệu** bằng cây cũ, để phép so sánh không lẫn hai biến):

| Cách làm | Accuracy | Thời gian | Số cây |
| --- | --- | --- | --- |
| **Cây tĩnh** (không đụng gì) | **0.5767** | **1.36 s** | 300 |
| Thay cây, dữ liệu gần đây (400 mẫu) | 0.5658 | 1.79 s | 300 |
| Thay cây, đủ dữ liệu như cây cũ | 0.5672 | 4.96 s | 300 |
| Chỉ thêm cây, không xoá | 0.5712 | 5.06 s | 390 |
| *Đoán bừa* | *0.5522* | — | — |

**Cả bốn phương án đều THUA cây tĩnh**, và tốn gấp gần bốn lần thời gian. Càng thay
đổi nhiều thì càng kém.

Giải thích: dữ liệu gần đây **ít hơn và nhiễu hơn**, nên cây mới yếu hơn cây cũ.
Việc thay cây chỉ có lợi khi thị trường **thật sự đổi chế độ** — mà bốn mã được chọn
lại cố ý là bốn mã **không có biến cố cấu trúc lớn** ([§3.2](#32-vì-sao-chọn-đúng-bốn-mã-này--một-quyết-định-có-kiểm-soát)).

> Thí nghiệm này được **giữ nguyên trong báo cáo** vì một kết quả âm được đo cẩn
> thận vẫn là một kết quả.

### 5.10.4. Thoái hoá theo thời gian — mô hình có hạn sử dụng

Phát lại toàn bộ dữ liệu sau tập huấn luyện, **từng phiên một**, đúng như mô hình sẽ
chạy ngoài thực tế: mỗi ngày dự đoán một lần, 5 phiên sau mới biết đúng hay sai.
**737 phiên được chấm**, chia đều thành năm giai đoạn liên tiếp:

| Giai đoạn | 2023-09→2024-04 | 2024-04→2024-11 | 2024-11→2025-06 | 2025-06→2026-01 | 2026-01→2026-08 |
| --- | --- | --- | --- | --- | --- |
| **Random Forest** | **0.6259** | **0.6463** | 0.5850 | 0.5102 | 0.5168 |
| Đoán bừa (majority) | 0.6259 | 0.5714 | 0.5782 | 0.4354 | 0.5503 |

Accuracy đi từ 0.63 xuống 0.52; **hai giai đoạn cuối gần như bằng đoán bừa**, và ở
giai đoạn cuối mô hình còn **thua** mốc đoán bừa.

Nghĩa là mô hình **có hạn sử dụng**, và biểu đồ này cho biết chu kỳ đó khoảng bao
lâu (**thoái hoá rõ từ giai đoạn thứ ba, tương ứng ~300 phiên**). Đây là thông tin
mà **một con số accuracy duy nhất không bao giờ nói ra được**.

## 5.11. Đối chiếu sáu tiêu chí chấp nhận

Quay lại đúng sáu dòng đã ghi ra ở [§2.4](#24-tiêu-chí-thành-công--đặt-ra-trước-khi-chạy-thí-nghiệm):

| # | Tiêu chí | Ngưỡng | Kết quả | Đạt |
| --- | --- | --- | --- | --- |
| 1 | Accuracy validate > 55% và cao hơn mọi đối chứng | > 0.55 | 0.6087 · đoán bừa 0.5788 · hồi quy 0.5353 | ✅ |
| 2 | ROC-AUC validate | > 0.55 | 0.6476 | ✅ |
| 3 | Chênh lệch train ↔ validate | < 0.10 | 0.0510 | ✅ |
| 4 | Độ lệch chuẩn walk-forward | < 0.05 | 0.0170 | ✅ |
| 5 | F1 của mọi lớp | ≥ 0.40 | 0.4098 và 0.7073 | ✅ |
| 6 | Test không suy giảm quá 5% so với validate | < 0.05 | **0.0850** | ❌ |

**Đạt 5/6.**

### Tự phê bình tiêu chí số 1

Về mặt câu chữ, tiêu chí 1 **đạt** — 0.6087 cao hơn cả đoán bừa lẫn hồi quy
logistic. Nhưng McNemar cho `p = 0.289`, nghĩa là chênh lệch đó **không phân biệt
được với may rủi**.

Điều này được ghi thẳng vào báo cáo, vì **nếu chỉ để dấu tick xanh thì đồ án đang
mắc đúng cái lỗi mà nó phê phán** ở [§5.3](#53-thí-nghiệm-rò-rỉ-có-kiểm-soát--vì-sao-có-báo-cáo-đạt-90).

> **Đánh giá trung thực:** đạt **5/6 tiêu chí về mặt hình thức**; về mặt thực chất,
> mô hình **chưa chứng minh được ưu thế đáng tin cậy so với đoán bừa**. Nếu làm
> lại, tiêu chí 1 nên được viết thành *"vượt đối chứng với `p < 0.05`"*.

## 5.12. Hiệu năng vận hành

Ngoài chất lượng dự đoán, hệ thống còn được đo về tốc độ:

| Hạng mục | Kết quả |
| --- | --- |
| Huấn luyện rừng 300 cây (bản đầu, `O(n²)`) | ~2 phút |
| Huấn luyện rừng 300 cây (sau tối ưu, `O(n log n)`) | **2.9 giây** — nhanh hơn ~40 lần |
| Một dự đoán trọn vẹn (60 dòng thô → 27 chỉ báo → 300 cây) | **1.7 ms** |
| Phát lại toàn bộ 763 phiên qua SSE | **1.2 giây** |
| Nạp gói mô hình từ file JSON | ~10 ms |
| Chạy toàn bộ 222 kiểm thử | 28.1 giây |
| Kích thước gói mô hình (`models/vnm.json`) | 754 KB |
| Ngữ cảnh build Docker (sau `.dockerignore`) | 710 MB → **4.7 MB** |
| Số gói phụ thuộc để chạy máy chủ | 9 → **4** |

Thị trường Việt Nam mỗi ngày một phiên, nên **về tốc độ hệ thống thừa sức chạy thời
gian thực**. Vấn đề không nằm ở tốc độ, mà ở độ chính xác.

---

# 6. CHỨC NĂNG CHÍNH CỦA HỆ THỐNG VÀ CÁCH SỬ DỤNG

## 6.1. Kiến trúc ba tầng

```text
  api/            FastAPI · SSE · dashboard Canvas 2D    ← biết về web
  ────────────────────────────────────────────────────
  src/pipeline/   chuỗi thời gian · chỉ báo · nhãn       ← biết về chứng khoán
  ────────────────────────────────────────────────────
  src/libraries/  cây · rừng · boosting · tuyến tính     ← CHỈ biết toán
```

**Ràng buộc quan trọng nhất: tầng dưới không bao giờ import tầng trên.**

Kèm theo là quy tắc *"không hằng số của đề tài trong mã nguồn"*: không file `.py`
nào trong `libraries/`, `pipeline/` hay `utilities/` được khai báo biến mang tên
hoặc giá trị cụ thể của đề tài. Tên biến phải thuộc về **thuật toán** (`series`,
`window`, `targets`, `impurity`), không thuộc về **dữ liệu** (`close_price`,
`vnm_volume`).

Mọi giá trị cụ thể chỉ xuất hiện ở hai nơi: **file cấu hình `config/*.json`** và
**notebook**. Nhờ vậy chuyển sang đề tài khác **không cần sửa một dòng mã nào**.

## 6.2. Bảy chức năng chính

| # | Chức năng | Điểm vào |
| --- | --- | --- |
| 1 | Huấn luyện + đánh giá đầu-cuối nhánh phân loại | `src/mainClassification.py` |
| 2 | Huấn luyện + đánh giá đầu-cuối nhánh hồi quy | `src/mainRegression.py` |
| 3 | Đóng gói mô hình ra file, dự đoán lại **không huấn luyện lại** | `--save-model` + `src/mainPredict.py` |
| 4 | Bảng điều khiển phát lại theo luồng thời gian thực | `api/main.py` (FastAPI + SSE) |
| 5 | Nạp dữ liệu người dùng và phát lại trên đó | Tab **Tải dữ liệu** của dashboard |
| 6 | Đóng gói triển khai bằng Docker | `Dockerfile` + `docker-compose.yml` |
| 7 | Khảo sát từng bước bằng notebook | `src/notebooks/01..05` |

---

## 6.3. Cài đặt

```powershell
# Clone và tạo môi trường ảo
git clone <repository-url>
cd BOT--Machine-Learning
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

Yêu cầu: **Python 3.10+** (phát triển trên 3.13) và Git.

| Nhóm | Gói | Vai trò |
| --- | --- | --- |
| Bắt buộc | `openpyxl`, `matplotlib`, `jupyter`, `notebook`, `ipykernel` | Đọc Excel, vẽ đồ thị, chạy notebook |
| Cho baseline | `numpy`, `pandas`, `scikit-learn` | **Chỉ** dùng trong `05_baseline_sklearn.ipynb` để đối chiếu |

> Tầng lõi `src/libraries/` **không phụ thuộc** bất kỳ thư viện ngoài nào.

## 6.4. Chức năng 1–2: Huấn luyện và đánh giá

```powershell
# Nhánh A — phân loại xu hướng tăng/giảm
python src/mainClassification.py --config config/vnm.json

# Nhánh B — hồi quy tỷ suất biến động
python src/mainRegression.py --config config/regression.json

# Bỏ qua bước vẽ nếu chỉ cần số liệu
python src/mainClassification.py --config config/vnm.json --no-figures
```

| Tham số | Ý nghĩa |
| --- | --- |
| `--config` *(bắt buộc)* | Đường dẫn file cấu hình JSON |
| `--no-figures` | Bỏ qua bước vẽ và lưu đồ thị |
| `--save-model ĐƯỜNG_DẪN` | Lưu gói mô hình ra file JSON (chỉ có ở nhánh A) |

**Đầu ra:** bộ chỉ số in ra màn hình + **12 đồ thị** ghi vào `data/output/` — phân
bố lớp, đường cong OOB, tầm quan trọng đặc trưng (MDI ↔ permutation), ma trận nhầm
lẫn, ROC, đường cong ngưỡng, walk-forward, và bốn đồ thị riêng cho nhánh hồi quy.

## 6.5. Chức năng 3: Gói mô hình và dự đoán lại

Đây là bước biến đồ án từ một **thuật toán** thành một **mô hình dùng lại được**.

```powershell
# Huấn luyện một lần rồi đóng gói ra file
python src/mainClassification.py --config config/vnm.json --save-model models/vnm.json

# Từ đó về sau chỉ nạp và dự đoán — mất dưới một giây
python src/mainPredict.py --model models/vnm.json --data data/input/VNM.csv --rows 10

# Chỉ xem dự báo mới nhất
python src/mainPredict.py --model models/vnm.json --data data/input/VNM.csv --rows 1

# Đổi ngưỡng quyết định
python src/mainPredict.py --model models/vnm.json --data data/input/VNM.csv --threshold 0.55

# Xem thông tin gói mô hình mà không chạy dự đoán
python src/mainPredict.py --model models/vnm.json --data data/input/VNM.csv --describe
```

### Gói mô hình gồm ba phần dính liền nhau

| Phần | Nội dung | Vì sao bắt buộc |
| --- | --- | --- |
| **Công thức** (`recipe`) | Bản đặc tả 27 đặc trưng, ánh xạ vai trò → tên cột, tham số làm sạch | Đặc trưng được *sinh ra*, không có sẵn trong dữ liệu. Thiếu phần này mô hình vẫn chạy nhưng **đọc nhầm chỉ báo — sai lặng lẽ, không báo lỗi** |
| **Tham số** (`model`) | Toàn bộ luật chia và giá trị tại lá của 300 cây | Chính là mô hình |
| **Siêu dữ liệu** | Nguồn dữ liệu, khoảng thời gian, số mẫu, tỷ lệ tham số/mẫu, chỉ số đạt được, ngưỡng quyết định | Để biết đang cầm mô hình nào và **tin được tới đâu** |

Ví dụ siêu dữ liệu thực tế trong `models/vnm.json`:

```json
{
  "training_summary": {
    "data_label": "VNM", "num_samples": 1719,
    "period": "2016-10-31 → 2023-09-06", "training_end": "2023-09-06",
    "horizon": 5, "parameters_per_sample": 11.44
  },
  "metrics": {
    "validation_accuracy": 0.6087, "validation_roc_auc": 0.6476,
    "test_accuracy": 0.5237, "out_of_bag_error": 0.3973
  },
  "size": {"num_trees": 300, "parameters": 19670},
  "threshold": 0.45
}
```

### Hai cơ chế an toàn

1. **Đối chiếu đặc trưng.** Trước khi dự đoán, hệ thống so tên **và thứ tự** bộ đặc
   trưng vừa dựng với bộ đặc trưng lúc huấn luyện. Chỉ cần hoán vị hai cột là chương
   trình dừng với thông báo rõ ràng, thay vì cho ra một con số vô nghĩa.
2. **Đường đi riêng cho dự đoán.** Lúc huấn luyện, các quan sát mới nhất bị loại vì
   chưa có nhãn. Lúc dự đoán thì **chính chúng mới là thứ ta cần**, nên
   `modelBundle.prepare_features_for_prediction()` vẫn dựng đặc trưng và làm sạch y
   hệt nhưng **giữ lại toàn bộ phần đuôi**. Dòng cuối được đánh dấu `← dự báo`.

### Dùng từ mã Python

```python
from pipeline.modelBundle import load_bundle, predict_with_bundle

bundle = load_bundle('models/vnm.json')      # ~10 ms
table  = {'Date': [...], 'Open': [...], 'High': [...],
          'Low': [...], 'Close': [...], 'Volume': [...]}
rows   = predict_with_bundle(bundle, table, num_rows=5)
```

**Yêu cầu dữ liệu đầu vào:** đúng tên cột `Date, Open, High, Low, Close, Volume`;
tối thiểu **50 dòng** (khuyến nghị ≥ 60) vì chỉ báo dài nhất cần 50 phiên khởi động;
sắp xếp theo thời gian tăng dần.

**Đọc kết quả cho đúng:** dự đoán ứng với **5 phiên sau**, không phải phiên kế tiếp.
Chi tiết và bảng tra lỗi: [`docs/predictionGuide.md`](docs/predictionGuide.md).

## 6.6. Chức năng 4: Bảng điều khiển phát lại

```powershell
python -m uvicorn api.main:application --port 8000
```

Mở `http://127.0.0.1:8000` — chọn mã, bấm **Bắt đầu**.

Màn hình phát lại từng phiên như một luồng trực tiếp, dự đoán tại mỗi bước và chấm
điểm ngay khi kết quả xuất hiện. **Đây không phải mẹo trình diễn** — nó chính là
kiểm định tiến dần ở [§5.5.2](#552-ổn-định-qua-kiểm-định-tiến-dần-walk-forward) được
trực quan hoá, cùng một phương pháp, chỉ khác ở chỗ kết quả hiện ra dần thay vì tổng
kết một lần ở cuối.

### Ba đường độ chính xác chạy song song

| Đường | Ý nghĩa |
| --- | --- |
| **Mô hình tĩnh** | Rừng đứng yên kể từ lúc huấn luyện |
| **Mô hình thích nghi** | Cùng rừng đó nhưng tiếp tục thay máu (`slidingForest.py`) |
| *Đoán bừa* | Luôn đoán lớp đa số của giai đoạn huấn luyện — **để mờ nhưng vẫn hiện** |

Đường thứ ba mới là đường khiến hai đường kia có ý nghĩa: nếu giấu nó đi thì hai
đường kia trông ấn tượng hơn thực tế rất nhiều.

### Ba điểm bảo đảm tính trung thực

- **Giai đoạn huấn luyện bị bỏ qua tự động.** Engine đọc mốc `training_end` trong
  gói mô hình và nhảy tới sau đó — phát lại trên dữ liệu mô hình đã học thuộc sẽ cho
  accuracy cao giả tạo.
- **Không nhìn trước.** Ở bước `t`, engine chỉ chạm dữ liệu tới `t`. Có kiểm thử
  riêng cắt chuỗi rồi chạy lại để bảo đảm điều này (`tests/test_replayEngine.py`).
- **Chấm điểm đúng độ trễ.** Với tầm nhìn 5 phiên, dự đoán tại `t` nằm trong **hàng
  đợi chờ** tới `t+5` mới được kiểm chứng. Chấm ngay chính là nhìn trộm tương lai.

### Các route API

| Route | Chức năng |
| --- | --- |
| `GET /` | Trang dashboard (`index.html` + `app.js` + `style.css`, không cần bước build) |
| `GET /api/models` | Liệt kê gói mô hình có trong `models/` |
| `GET /api/datasets` | Liệt kê dữ liệu có trong `data/input/` |
| `POST /api/upload` | Nhận file `.csv` / `.json` / `.xlsx` người dùng tải lên |
| `GET /api/stream` | Luồng Server-Sent Events phát lại từng phiên |

Biểu đồ nến được vẽ **trực tiếp bằng Canvas 2D**, không dùng thư viện biểu đồ nào.

## 6.7. Chức năng 5: Nạp dữ liệu người dùng

Tab **Tải dữ liệu** nhận file `.csv` / `.json` / `.xlsx` và chạy qua **đúng cùng một
bộ máy phát lại** — không có đường xử lý song song thứ hai, vì hai đường song song
thì sớm muộn cũng lệch nhau.

Phần kiểm tra đầu vào báo lỗi **cụ thể**: thiếu cột nào, file đang có cột nào, cần
tối thiểu bao nhiêu dòng — thay vì ném ra một traceback Python khó hiểu.

> Thử mô hình của mã này trên dữ liệu mã khác là việc **nên làm** — đó chính là cách
> tự kiểm chứng xem tín hiệu có chung giữa các mã hay không
> ([§4.2](#42-lựa-chọn-tầm-nhìn-dự-báo) cho thấy câu trả lời là **không**).

## 6.8. Chức năng 6: Đóng gói Docker

```powershell
# Huấn luyện trước để có gói mô hình
python src/mainClassification.py --config config/vcb.json --no-figures --save-model models/vcb.json

# Dựng và chạy
docker compose up --build
```

Mở `http://localhost:8000`.

**Ảnh chỉ chứa phần cần để chạy máy chủ.** Nhóm phân tích — `matplotlib`, `jupyter`,
`numpy`, `pandas`, `scikit-learn` — bị bỏ hẳn ra ngoài, vì lõi học máy trong `src/`
là Python thuần và không đụng tới chúng:

| | Gói trực tiếp | Ngữ cảnh build |
| --- | --- | --- |
| `requirements.txt` (đầy đủ) | 9 | 710 MB |
| `requirements-api.txt` (chạy máy chủ) | **4** | **4.7 MB** |

Bốn gói đó là `fastapi`, `uvicorn`, `python-multipart` và `openpyxl` (gói cuối chỉ
dùng khi người dùng tải lên file `.xlsx`).

`docker-compose.yml` **gắn** `models/` và `data/input/` từ máy chủ nhà vào container
ở chế độ **chỉ đọc**, thay vì nướng cứng vào ảnh — huấn luyện lại trên máy là
container thấy ngay, không phải build lại. Nếu `models/` còn rỗng, container **vẫn
chạy bình thường**: trang chính phục vụ được, healthcheck vẫn xanh, và `/api/models`
trả về đúng câu hướng dẫn phải huấn luyện trước.

## 6.9. Chức năng 7: Notebook khảo sát

```powershell
jupyter notebook src/notebooks/
```

| Notebook | Nội dung |
| --- | --- |
| `01_eda.ipynb` | Khảo sát dữ liệu, phân bố lớp, diễn biến giá |
| `02_features.ipynb` | Chỉ báo kỹ thuật + **kiểm tra tính nhân quả** |
| `03_train_classify.ipynb` | Nhánh A đầy đủ, có phân tích quá khớp |
| `04_train_regress.ipynb` | Nhánh B, minh hoạ giới hạn không ngoại suy |
| `05_baseline_sklearn.ipynb` | Đối chiếu `scikit-learn` + Gradient Boosting + `GridSearchCV` |

## 6.10. Chạy kiểm thử

```powershell
python -m pytest tests/ -q
# → 222 passed
```

| File | Kiểm tra |
| --- | --- |
| `test_indicators.py` | Nhân quả, giữ độ dài, đúng công thức |
| `test_metrics.py` | Đối chiếu giá trị tính tay |
| `test_forest.py` | Gini/Entropy, bootstrap ≈ `1/e`, tái lập |
| `test_linearModel.py` | Chuẩn hoá, hội tụ, đối chiếu `scikit-learn` |
| `test_modelStore.py` | Lưu-nạp giữ nguyên dự đoán, chặn lệch đặc trưng |
| `test_slidingForest.py` | Cây luân chuyển, rừng gốc không bị đổi |
| `test_replayEngine.py` | **KHÔNG NHÌN TRƯỚC**, hàng đợi chờ đúng độ trễ |
| `test_api.py` | Đọc file tải lên, khung SSE, các route |

## 6.11. Đổi sang mã cổ phiếu hoặc đề tài khác

**Không cần sửa một dòng mã nào** — chỉ sửa `config/*.json`:

| Muốn đổi | Sửa khoá |
| --- | --- |
| Nguồn dữ liệu | `dataset.path`, `dataset.label` |
| Tên cột trong file | `dataset.series` (ánh xạ vai trò → tên cột thật) |
| Bộ chỉ báo và cửa sổ | `features` (danh sách bản đặc tả) |
| Tầm nhìn dự báo | `labeling.horizon` |
| Tỷ lệ tách tập và gap | `split` |
| Siêu tham số mô hình | `model` |

Cơ chế `extends` cho phép tách **phương pháp** khỏi **dữ liệu**: `config/base.json`
giữ toàn bộ bản đặc tả 27 đặc trưng và siêu tham số dùng chung; bốn file
`vnm/vcb/msn/mwg.json` chỉ khai báo đường dẫn dữ liệu và tiền tố tên file đầu ra.
**Sửa phương pháp ở một chỗ là sửa cho cả bốn mã** — đó chính là điều kiện để phép
so sánh giữa bốn mã có ý nghĩa.

Danh sách chỉ báo hợp lệ lấy bằng `pipeline.featureBuilder.describe_registry()`.

---

# 7. KHÓ KHĂN GẶP PHẢI VÀ CÁCH KHẮC PHỤC

## 7.1. Bảng tổng hợp

| # | Khó khăn | Nhóm | Cách khắc phục | Kết quả |
| --- | --- | --- | --- | --- |
| 1 | Huấn luyện quá chậm (~2 phút/lần) | Hiệu năng | Bộ tích luỹ quét ngưỡng `O(n log n)` | **2.9 s** (×40) |
| 2 | Rò rỉ nhãn do dịch đặc trưng sai chiều | Đúng đắn | Dịch nhãn về sau, bỏ 5 dòng cuối | Accuracy ảo 0.81 → thật 0.56 |
| 3 | Rò rỉ qua đường biên tách tập | Đúng đắn | `gap = horizon = 5` ở mỗi ranh giới | Loại bỏ 5 mẫu rò rỉ/biên |
| 4 | 355 phiên "đứng yên" gây mất cân bằng giả | Dữ liệu | `flat_label = null` — loại khỏi tập học | Phân bố 59/41 → 53/47 |
| 5 | Quá khớp nghiêm trọng (train 1.0000) | Mô hình | Siết `max_depth = 5`, `min_samples_leaf = 40` | Chênh 0.4293 → **0.0347** |
| 6 | MDI xếp hạng sai do đặc trưng tương quan | Diễn giải | Bổ sung permutation importance | Phát hiện 25/27 đặc trưng dư thừa |
| 7 | Mô hình lưu ra file dự đoán sai lặng lẽ | Kiến trúc | Gói **công thức + tham số + siêu dữ liệu** đi liền | Lệch đặc trưng → dừng có thông báo |
| 8 | Dashboard vô tình "nhìn trước" | Đúng đắn | Hàng đợi chờ `t+5` + bỏ qua giai đoạn train | Có test tự động chặn tái phát |
| 9 | Ảnh Docker phình 710 MB | Triển khai | Tách `requirements-api.txt` + `.dockerignore` | **4.7 MB**, 4 gói |
| 10 | Rừng thích nghi không cải thiện được gì | Nghiên cứu | Đo 4 phương án, giữ nguyên kết quả âm | Xác lập: cây tĩnh tốt hơn |
| 11 | Không biết chọn tầm nhìn nào | Thiết kế | Đo cả `h = 1` và `h = 5` trên 2 mã | Chọn `h = 5` có căn cứ |
| 12 | Thiên lệch do dò tham số trên một mã | Phương pháp | Thêm 3 mã **không** chạm tham số | Đo được **độ lớn** thiên lệch |
| 13 | Không biết kết quả có ý nghĩa thống kê không | Đánh giá | Kiểm định McNemar cho cả 4 mã | Phát hiện `p = 0.289` |
| 14 | Không chứng minh được quy trình sạch | Phương pháp | Phép thử xáo nhãn + tái hiện rò rỉ | AUC sụp về ~0.45 khi xáo nhãn |

---

## 7.2. Phân tích chi tiết các khó khăn tiêu biểu

### 7.2.1. Hiệu năng huấn luyện — nút thắt chặn cả tiến độ nghiên cứu

**Vấn đề.** Bản cài đặt đầu tiên tìm ngưỡng chia bằng cách: với mỗi ngưỡng ứng
viên, quét lại toàn bộ mẫu để đếm số nhãn mỗi bên. Với `n` mẫu và `n` ngưỡng ứng
viên, chi phí mỗi nút là `O(n²)`. Huấn luyện 300 cây mất **khoảng 2 phút**.

**Vì sao đây là vấn đề nghiêm trọng.** Không phải vì người dùng phải chờ, mà vì
**nó chặn nghiên cứu**: đồ án cần chạy hàng chục thí nghiệm (bốn mã × hai nhánh,
quét ràng buộc, quét kích thước mô hình, đường cong học 5 điểm, xáo nhãn 3 lần, bốn
phương án thích nghi). Với 2 phút mỗi lần chạy, phần lớn số liệu trong
[§5](#5-kết-quả-thí-nghiệm) sẽ **không tồn tại**.

**Cách khắc phục.** Thay thế bằng **một lượt quét duy nhất** với bộ tích luỹ
(`LabelSplitAccumulator` và `ValueSplitAccumulator` trong `rfMath.py`): sắp xếp mẫu
theo giá trị đặc trưng một lần, rồi vừa đi vừa **cộng dồn** số đếm mỗi lớp sang
trái, cập nhật `O(1)` mỗi bước. Chi phí giảm còn `O(n log n)` do chi phí sắp xếp.

**Kết quả.** Từ ~2 phút xuống **2.9 giây — nhanh hơn khoảng 40 lần, kết quả giống
hệt từng chữ số**. Tính "giống hệt" được bảo đảm bằng kiểm thử đối chiếu.

### 7.2.2. Rò rỉ dữ liệu — lỗi nguy hiểm nhất vì nó *cải thiện* kết quả

**Vấn đề.** Rò rỉ dữ liệu không làm chương trình báo lỗi; nó làm **kết quả đẹp lên**.
Đây là lý do nó nguy hiểm — không có tín hiệu nào cảnh báo, và người làm có xu hướng
tin vào con số có lợi cho mình.

Đồ án gặp rò rỉ ở **hai** chỗ khác nhau:

**(a) Dịch đặc trưng thay vì dịch nhãn.** Cách cài đặt ban đầu dịch cột đặc trưng
lùi một phiên, khiến chỉ báo tại `t` chứa thông tin của phiên `t+1`. Hệ quả:
Accuracy validate **tăng ảo từ 0.56 lên 0.81**.

*Khắc phục:* giữ nguyên đặc trưng tại `t`, lấy nhãn từ `Close(t+h)`, bỏ `h` dòng
cuối do không có nhãn. Ghi lại nguyên nhân trong commit
`fix(labeling): dịch nhãn về sau thay vì dịch đặc trưng về trước`.

**(b) Rò rỉ qua đường biên tách tập.** Tinh vi hơn nhiều: nhãn của mẫu cuối tập
train được tính từ giá 5 phiên sau đó — tức đã **chạm vào vùng validate**. Chỉ 5
mẫu, nhưng vẫn là rò rỉ.

*Khắc phục:* chèn `gap = horizon` phiên ở mỗi ranh giới (`split.gap = 5`).

**Cách xác minh không còn rò rỉ nào khác.** Vấn đề của việc chống rò rỉ là **không
thể chứng minh bằng cách đọc mã** — luôn còn khả năng bỏ sót. Đồ án giải quyết bằng
hai thí nghiệm:

1. **Phép thử xáo nhãn** ([§5.2](#52-phép-thử-xáo-nhãn--kiểm-tra-rò-rỉ)) — nếu còn
   đường tắt nào, mô hình sẽ tìm ra nó ngay cả khi nhãn bị phá sạch. AUC sụp về
   0.41–0.54 chứng tỏ không có.
2. **Kiểm thử tính nhân quả tự động** — cắt chuỗi tại vị trí 1 000, tính lại chỉ
   báo trên đoạn đầu, so từng giá trị với chuỗi đầy đủ. Chạy trên mọi chỉ báo, mọi
   lần commit.

### 7.2.3. Quan sát "đứng yên" — bẫy tiền xử lý

**Vấn đề.** Ở tầm nhìn 1 phiên, 355/2 587 phiên (13.7%) có giá đóng cửa **không
đổi** do bước giá làm tròn của sàn. Nhãn nhị phân `tăng/giảm` không có chỗ cho nhóm
này.

**Cái bẫy.** Cách xử lý "tự nhiên" nhất — dồn chúng vào lớp "giảm" (vì `không tăng`)
— tạo ra phân bố **59.2 / 40.8**. Mô hình học được điều tối ưu về mặt toán học:
**luôn đoán "giảm"**. Accuracy đạt 0.59 nghe khá, nhưng F1 của lớp "tăng" chỉ
**0.1878** — mô hình đã bỏ rơi hoàn toàn một lớp.

**Khắc phục.** Loại nhóm đứng yên khỏi tập huấn luyện (`flat_label = null`). Phân bố
trở về **52.7 / 47.3**, mốc majority từ 0.5916 xuống 0.5269, và mô hình **buộc phải
học tín hiệu thật** thay vì học tỷ lệ lớp.

**Bài học rút ra:** tác động của quyết định tiền xử lý này (0.59 ↔ 0.53) **lớn hơn
tác động của mọi phép tinh chỉnh siêu tham số cộng lại**.

### 7.2.4. Quá khớp — và nghịch lý "siết chặt lại thì tốt lên"

**Vấn đề.** Cấu hình mặc định (`max_depth = None`, `min_samples_leaf = 1`) cho
Accuracy train **1.0000** — ghi nhớ hoàn hảo — nhưng validate chỉ 0.5707, **thấp
nhất trong mọi cấu hình thử nghiệm**.

**Khắc phục.** Quét ràng buộc có hệ thống
([§5.4](#54-vì-sao-phải-ràng-buộc-mô-hình)), chọn `max_depth = 5`,
`min_samples_leaf = 40`. Chênh lệch train↔validate giảm từ **0.4293 xuống 0.0347**,
**đồng thời** ROC-AUC validate tăng từ 0.6066 lên 0.6418.

**Điều đáng chú ý về mặt nhận thức:** hai chỉ số này đi **cùng chiều**, không phải
đánh đổi. Với dữ liệu nhiễu như giá cổ phiếu, **mọi tham số dư thừa đều bị dùng để
học thuộc nhiễu**. Phát hiện này còn được đẩy xa hơn ở
[§5.10.2](#5102-kích-thước-mô-hình--đang-thừa-tham-số-gấp-10-lần): ngay cả cấu hình
đã siết vẫn còn **thừa tham số gấp 10 lần**.

### 7.2.5. Diễn giải sai vì đặc trưng tương quan

**Vấn đề.** Bảng MDI xếp `atr_over_close` hạng 2 với trọng số 0.1044 — có vẻ là một
đặc trưng quan trọng. Nếu chỉ nhìn MDI, kết luận sẽ là *"ATR là chỉ báo giá trị"*.

**Phát hiện.** Permutation importance của nó là **−0.00326** — xáo trộn nó còn làm
mô hình **tốt lên**. Nguyên nhân: `atr_over_close`, `volatility_10`, `volatility_20`
và `bollinger_width` **cùng đo một thứ**. Khi phá hỏng một cái, ba cái còn lại vẫn
cung cấp đủ thông tin. MDI **chia đều công trạng** cho cả nhóm và tạo ảo giác rằng
cả bốn đều quan trọng.

**Khắc phục.** Luôn báo cáo **hai** thước đo cạnh nhau, và khi chúng bất đồng thì
**tin permutation** (đo trên tập validate, đo *mất mát thật* thay vì *mức độ được
dùng*).

**Hệ quả tìm được:** chỉ **2 trong 27** đặc trưng thực sự không thay thế được.

### 7.2.6. Mô hình lưu ra file có thể "sai lặng lẽ"

**Vấn đề.** Đặc trưng trong đồ án **được sinh ra**, không có sẵn trong dữ liệu. Nếu
chỉ lưu tham số cây mà không lưu công thức sinh đặc trưng, thì khi nạp lại và dựng
đặc trưng theo một thứ tự khác, mô hình **vẫn chạy và vẫn cho ra số** — nhưng cây
đang đọc `rsi_14` ở vị trí mà thực tế là `volatility_20`. **Sai hoàn toàn, không có
lỗi nào được ném ra.**

**Khắc phục.** Gói mô hình buộc **ba phần đi liền nhau, không tách rời được**: công
thức + tham số + siêu dữ liệu ([§6.5](#65-chức-năng-3-gói-mô-hình-và-dự-đoán-lại)).
Trước mỗi lần dự đoán, `verify_feature_compatibility()` so **tên và thứ tự** — chỉ
cần hoán vị hai cột là chương trình dừng với thông báo rõ ràng.

**Điều cố ý không lưu:** tập mẫu huấn luyện và chi tiết out-of-bag — chúng chiếm
phần lớn dung lượng mà chỉ phục vụ việc đánh giá lúc huấn luyện; con số OOB đã được
ghi sẵn trong siêu dữ liệu nên không mất thông tin.

### 7.2.7. Dashboard suýt "nhìn trước" theo hai cách

Bảng điều khiển phát lại là nơi rò rỉ **dễ tái xuất hiện nhất**, vì lúc chạy thật
mọi dữ liệu đều đã nằm sẵn trong bộ nhớ.

| Nguy cơ | Hệ quả nếu bỏ sót | Cách chặn |
| --- | --- | --- |
| Phát lại từ đầu chuỗi | Accuracy cao giả tạo trên phần mô hình đã học thuộc | Đọc `training_end` trong gói mô hình, nhảy tới sau đó |
| Chấm điểm ngay tại `t` | Dùng giá `t+5` — chính là nhìn trộm tương lai | **Hàng đợi chờ**: dự đoán tại `t` chỉ được chấm ở `t+5` |
| Dựng đặc trưng trên toàn chuỗi | Chỉ báo tại `t` nhiễm dữ liệu tương lai | Ở bước `t`, engine **chỉ chạm dữ liệu tới `t`** |

Cả ba đều có kiểm thử tự động trong `tests/test_replayEngine.py` — cắt chuỗi rồi
chạy lại để bảo đảm kết quả không đổi.

### 7.2.8. Thiên lệch chọn mô hình — vấn đề không có cách "sửa", chỉ có cách đo

**Vấn đề.** Siêu tham số được dò trên VNM, rồi kết quả VNM lại được báo cáo. Điều
này khiến con số 0.6087 **lạc quan hơn thực tế** — nhưng không biết lạc quan bao
nhiêu.

**Khắc phục.** Không thể "sửa" bằng cách bỏ VNM đi (như vậy là giấu số liệu), nên
đồ án chọn cách **đo độ lớn của thiên lệch**: thêm ba mã VCB, MSN, MWG chạy với
**đúng bộ siêu tham số đó, không được chạm vào một lần nào**.

**Kết quả:** cả ba mã đều cho Accuracy validate thấp hơn VNM (0.5359 / 0.5669 /
0.4955 so với 0.6087) — **đúng như dự đoán nếu thiên lệch tồn tại**. Việc ghi rõ
điều này trong `config/vnm.json` giúp người đọc mã nguồn biết ngay con số nào cần
đọc thận trọng.

### 7.2.9. Không biết kết quả có ý nghĩa thống kê hay không

**Vấn đề.** Accuracy 0.6087 so với đoán bừa 0.5788 — chênh 3 điểm phần trăm trên 368
mẫu. **Chênh lệch này có phân biệt được với may rủi không?** Bản thân con số accuracy
không trả lời được.

**Khắc phục.** Áp dụng **kiểm định McNemar** — chỉ xét những mẫu mà hai mô hình bất
đồng, vì phần chúng giống nhau không phân biệt được chúng.

**Kết quả gây thất vọng nhưng quan trọng:** `p = 0.289` — **không đạt** mức ý nghĩa
5%. Đồ án giữ nguyên kết luận này và ghi thẳng vào phần tự phê bình tiêu chí 1
([§5.11](#511-đối-chiếu-sáu-tiêu-chí-chấp-nhận)), thay vì lặng lẽ bỏ qua kiểm định.

---

# 8. TRANH LUẬN, KHÁM PHÁ VÀ KẾT LUẬN

## 8.1. Năm khám phá chính

### Khám phá 1 — Tiền xử lý quyết định kết quả nhiều hơn mô hình

Việc xử lý 355 quan sát đứng yên làm thay đổi mốc majority từ 0.59 xuống 0.53. Để so
sánh: dò `GridSearchCV` trên **60 cấu hình** siêu tham số chỉ cải thiện Accuracy
validate từ 0.6087 lên 0.6168 — **0.8 điểm phần trăm**.

> Nói cách khác: **một dòng cấu hình tiền xử lý có tác động gấp bảy lần toàn bộ công
> sức tinh chỉnh siêu tham số.** Đây là điều ngược với trực giác thông thường —
> phần lớn thời gian của người mới thường dồn vào chọn mô hình và tinh chỉnh tham số.

### Khám phá 2 — Ràng buộc mô hình chặt lại làm kết quả tốt lên

Thông thường ta hình dung có sự **đánh đổi** giữa độ khớp tập huấn luyện và khả năng
tổng quát hoá. Với dữ liệu tài chính, quan hệ này **không phải đánh đổi mà là cùng
chiều**: siết ràng buộc vừa giảm khoảng cách train↔validate (0.4293 → 0.0347) vừa
**tăng** ROC-AUC validate (0.6066 → 0.6418).

Kết quả này còn được đẩy xa hơn: mô hình **1 955 tham số** cho AUC tương đương và
Accuracy test **cao hơn** mô hình 19 670 tham số. Khoảng **90% dung lượng mô hình
hiện tại đang phục vụ việc học thuộc nhiễu**.

### Khám phá 3 — Chênh lệch giữa hai giai đoạn lớn gấp ba lần chênh lệch giữa các mô hình

| Nguồn chênh lệch | Độ lớn |
| --- | --- |
| Giữa **năm mô hình khác nhau** (hai họ ensemble, hai cài đặt, một tuyến tính) trên cùng tập validate | < 3 điểm phần trăm |
| Của **cùng một mô hình** giữa validate và test | **8.5 điểm phần trăm** |

Bằng chứng mạnh nhất là **thứ hạng giữa các mô hình đảo ngược hoàn toàn**: Logistic
Regression tệ nhất trên validate (AUC 0.5081 — không phân biệt được gì) nhưng tốt
nhất trên test (AUC 0.6086); trên bốn mã, Random Forest thắng 3/4 ở validate nhưng
chỉ 1/4 ở test.

**Nếu hai giai đoạn cùng phân phối, điều này không thể xảy ra.** Việc thứ hạng đảo
ngược chỉ có thể giải thích bằng việc **quan hệ giữa đặc trưng và mục tiêu đã thay
đổi** giữa hai giai đoạn.

> **Hệ quả về mặt phương pháp:** công sức nên dồn vào **thiết kế quy trình kiểm định
> trung thực**, không phải vào việc chọn mô hình. Đây cũng là câu trả lời cho câu
> hỏi *"sao không dùng LSTM/Transformer?"* — khi mô hình tuyến tính đơn giản nhất
> còn chưa thua rõ ràng thì thêm độ phức tạp chỉ làm tăng phương sai.

### Khám phá 4 — Dữ liệu này cho phép đoán *hướng*, không cho phép đoán *độ lớn*

`R²` trên validate: chỉ VNM dương (+0.0716); **ba trên bốn mã cho `R²` âm**, tức
**thua cả việc đoán bừa bằng trung bình tập huấn luyện**.

Nguyên nhân mang tính cấu trúc, không phải lỗi cài đặt: rừng lấy trung bình 300 cây,
mỗi cây lấy trung bình các giá trị trong lá — **hai lớp lấy trung bình chồng lên
nhau san phẳng mọi giá trị cực đoan**. Biên độ dự đoán hẹp hơn biên độ thực tế
**4–7 lần**.

Đồng thời, `Directional Accuracy` của nhánh B (0.6137) gần như trùng nhánh A
(0.6087). Nghĩa là **bắt mô hình học thêm độ lớn không giúp nó đoán hướng tốt hơn**
— phần thông tin dùng được nằm ở **dấu**, còn độ lớn hầu như chỉ là nhiễu.

> Đây là **căn cứ định lượng** cho quyết định đóng khung đề tài thành bài toán phân
> loại, thay vì một lựa chọn theo cảm tính.

### Khám phá 5 — Mô hình đang bị *thiếu dữ liệu*, không phải *thiếu độ phức tạp*

Đường cong học vẫn **dốc lên ở điểm cuối** (AUC 0.4496 → 0.6418 khi tăng từ 343 lên
1 719 mẫu, chưa chạm trần). Trong khi đó, giảm số tham số đi 10 lần **không làm kết
quả xấu đi**.

Hai quan sát này cùng chỉ về một hướng: **hướng cải thiện đúng là thêm dữ liệu (đa
mã, chuỗi dài hơn), không phải thêm độ phức tạp cho mô hình.**

## 8.2. Tranh luận: nên đánh giá đồ án này như thế nào?

### Lập luận phản biện 1: *"Accuracy 60% là quá thấp, nhiều báo cáo khác đạt 90%"*

**Trả lời bằng số liệu.** Đồ án đã tái hiện chính xác cơ chế tạo ra những con số cao
đó ([§5.3](#53-thí-nghiệm-rò-rỉ-có-kiểm-soát--vì-sao-có-báo-cáo-đạt-90)): cùng dữ
liệu, cùng thuật toán, chỉ đổi cách tách tập thì được 0.5718 → 0.6612 → **1.0000**.

Con số 0.5292 và con số 0.90 **không so sánh được với nhau vì chúng đo hai thứ khác
nhau**: một cái đo khả năng dự đoán tương lai, một cái đo khả năng nhận ra hàng xóm
hoặc ghi nhớ dữ liệu đã học.

Hệ thống có thể tạo ra con số 0.90 trong vòng năm phút. Nó chọn không làm vậy, và
**chứng minh được rằng đó là lựa chọn chứ không phải giới hạn năng lực**.

### Lập luận phản biện 2: *"5/6 tiêu chí đạt — vậy đồ án thành công?"*

**Không hoàn toàn, và đồ án tự nói ra điều đó.** Tiêu chí 1 đạt về mặt câu chữ nhưng
McNemar cho `p = 0.289` — chênh lệch **không phân biệt được với may rủi**. Nếu chỉ
để dấu tick xanh thì đồ án đang mắc đúng cái lỗi mà nó phê phán.

Đánh giá đúng là: **5/6 về mặt hình thức, nhưng chưa chứng minh được ưu thế đáng tin
cậy so với đoán bừa.** Bản thân việc phát hiện được điều này — bằng cách chủ động
chạy kiểm định thống kê thay vì dừng ở accuracy — mới là phần có giá trị.

### Lập luận phản biện 3: *"Sao không dùng scikit-learn cho nhanh?"*

Ngoài yêu cầu của đề bài về việc hiểu thuật toán, việc tự cài đặt mang lại thứ mà
gọi thư viện không cho: khi thấy tỷ lệ out-of-bag khớp `1/e` đến bốn chữ số, ta
**biết chắc** phần bootstrap của mình đúng — chứ không phải tin rằng nó đúng.

`scikit-learn` **vẫn được dùng** — ở vai trò **kiểm chứng**
([§5.1b](#b-đối-chiếu-với-scikit-learn)), không phải ở vai trò chạy.

### Lập luận phản biện 4: *"Kết quả âm của rừng thích nghi có nên đưa vào báo cáo?"*

**Có.** Một kết quả âm được đo cẩn thận vẫn là một kết quả. Thí nghiệm này bác bỏ
một giả thuyết hợp lý (*"cho mô hình học thêm dữ liệu mới thì nó sẽ tốt hơn"*) bằng
bốn phương án đo có kiểm soát, kể cả phương án công bằng nhất về lượng dữ liệu.

Việc giữ nó lại cũng là một phép thử về tính nhất quán: một báo cáo chỉ trình bày
những thí nghiệm thành công thì bản thân nó **đã là một dạng chọn lọc thiên lệch**.

## 8.3. Kết luận

### Đã làm được

1. **Quy trình học máy đầy đủ** — 8 bước từ nạp dữ liệu tới báo cáo, mỗi bước là một
   module riêng có thể kiểm thử độc lập.
2. **Thuật toán tự cài đặt hoàn toàn** — Decision Tree, Random Forest, Gradient
   Boosting, hồi quy tuyến tính và logistic viết thuần Python, kiểm chứng bằng ba
   cách độc lập (`1/e`, đối chiếu `scikit-learn`, 222 test).
3. **Chống rò rỉ bốn lớp, và chứng minh được là sạch** — bằng phép thử xáo nhãn, chứ
   không bằng lời khẳng định.
4. **Bốn mã, một bộ siêu tham số, có đối chứng tuyến tính và kiểm định thống kê** —
   thiết kế thí nghiệm cho phép đo được cả độ lớn của thiên lệch chọn mô hình.
5. **Hệ thống chạy thật** — máy chủ dự đoán thời gian thực với dashboard phát lại,
   nạp dữ liệu ngoài, đóng gói Docker 4.7 MB.
6. **Đánh giá trung thực** — báo cáo cả phần không đạt kèm phân tích nguyên nhân,
   thay vì chỉ trình bày con số đẹp.

### Chưa làm được — giới hạn được tuyên bố rõ

| Giới hạn | Bằng chứng |
| --- | --- |
| Ưu thế so với đoán bừa **chưa đạt ý nghĩa thống kê** ở bất kỳ mã nào | McNemar `p = 0.289` (VNM) — [§5.6](#56-bốn-mã-một-bộ-siêu-tham-số--kiểm-định-thống-kê-mcnemar) |
| Mô hình **thoái hoá rõ từ giai đoạn thứ ba** (~300 phiên) | 0.6259 → 0.5168 — [§5.10.4](#5104-thoái-hoá-theo-thời-gian--mô-hình-có-hạn-sử-dụng) |
| Không dự báo được biến động mạnh | Biên độ dự đoán hẹp hơn thực tế 4–7 lần — [§5.7.3](#573-giới-hạn-không-ngoại-suy--quan-sát-trực-tiếp) |
| Dữ liệu còn thiếu | Đường cong học chưa chạm trần — [§5.10.1](#5101-đường-cong-học--mô-hình-còn-thiếu-dữ-liệu) |
| Mô hình thừa tham số gấp 10 lần | 1 955 tham số cho kết quả tốt hơn 19 670 — [§5.10.2](#5102-kích-thước-mô-hình--đang-thừa-tham-số-gấp-10-lần) |
| Chưa tính phí giao dịch, trượt giá, thanh khoản | Ngoài phạm vi — **không dùng để đầu tư** |
| Giả định tính dừng bị vi phạm | Thứ hạng mô hình đảo ngược giữa hai giai đoạn — [§5.8](#58-so-sánh-các-họ-mô-hình) |

### Kết luận cuối cùng

> **Hệ thống này chứng minh được quy trình, chưa chứng minh được lợi nhuận.**

Và đó là kết quả đúng. Nếu một đồ án sinh viên chạy trong vài tuần mà dự đoán được
thị trường chứng khoán, thì khả năng cao nhất là **quy trình có chỗ sai, chứ không
phải thị trường có chỗ hở**.

Giá trị thực chất của đồ án nằm ở chỗ nó **đo được chính xác mình yếu ở đâu** —
thoái hoá sau bao lâu, thiếu bao nhiêu dữ liệu, thừa bao nhiêu tham số, ưu thế có ý
nghĩa thống kê hay không — thay vì dừng lại ở một con số accuracy đơn lẻ không nói
lên điều gì.

---

# 9. HƯỚNG PHÁT TRIỂN TRONG TƯƠNG LAI

Các hướng dưới đây được sắp theo **mức độ được số liệu của đồ án ủng hộ**, từ mạnh
đến yếu — không phải theo mức độ hấp dẫn về mặt kỹ thuật.

## 9.1. Ưu tiên cao — có bằng chứng trực tiếp từ thí nghiệm

| # | Hướng | Bằng chứng ủng hộ | Cách làm |
| --- | --- | --- | --- |
| 1 | **Huấn luyện đa mã** | [§5.10.1](#5101-đường-cong-học--mô-hình-còn-thiếu-dữ-liệu): đường cong học chưa chạm trần | Gộp nhiều mã thành một tập huấn luyện chung; đặc trưng đã ở dạng tỷ lệ nên so sánh được giữa các mã |
| 2 | **Huấn luyện lại theo cửa sổ trượt** | [§5.10.4](#5104-thoái-hoá-theo-thời-gian--mô-hình-có-hạn-sử-dụng): thoái hoá rõ sau ~300 phiên | `splitter.generate_rolling_window_folds()` đã sẵn sàng; cần xác lập chu kỳ tối ưu bằng thực nghiệm |
| 3 | **Rút gọn mô hình xuống ~2 000 tham số** | [§5.10.2](#5102-kích-thước-mô-hình--đang-thừa-tham-số-gấp-10-lần): bản nhỏ cho test tốt hơn | Dò lại từ đầu với lưới nghiêng về mô hình nhỏ, **quyết định trên validate** để tránh lặp lại lỗi chọn theo test |
| 4 | **Rút gọn bộ đặc trưng** | [§5.9](#59-diễn-giải-mô-hình--đặc-trưng-nào-thực-sự-được-dùng): chỉ 2/27 đặc trưng không thay thế được | Chọn lọc bằng permutation importance đo trên **tập train nội bộ**, không dùng validate |

## 9.2. Ưu tiên trung bình — hợp lý về mặt bài toán

| # | Hướng | Lý do |
| --- | --- | --- |
| 5 | **Nhãn ba lớp** (`tăng / đi ngang / giảm`) với ngưỡng theo `ATR` | Không bắt mô hình phán bừa khi thị trường đi ngang; `labeling.create_ternary_labels()` đã có sẵn |
| 6 | **Sửa lại tiêu chí chấp nhận số 1** thành *"vượt đối chứng với `p < 0.05`"* | [§5.11](#511-đối-chiếu-sáu-tiêu-chí-chấp-nhận): tiêu chí hiện tại đạt về câu chữ nhưng không đạt về thực chất |
| 7 | **Backtest có tính phí giao dịch và trượt giá** | Accuracy 0.52 rất có thể **không sinh lời** sau phí — cần đo để biết chắc |
| 8 | **Bổ sung đặc trưng ngoài phân tích kỹ thuật** | Tin tức, báo cáo tài chính, dòng tiền khối ngoại, chỉ số vĩ mô — mở rộng nguồn tín hiệu thay vì khai thác kỹ hơn nguồn đã cạn |

## 9.3. Ưu tiên thấp — nên thận trọng

| # | Hướng | Vì sao thận trọng |
| --- | --- | --- |
| 9 | **XGBoost / LightGBM** | [§8.1](#khám-phá-3--chênh-lệch-giữa-hai-giai-đoạn-lớn-gấp-ba-lần-chênh-lệch-giữa-các-mô-hình): chênh lệch giữa các họ mô hình < 3 điểm, nhỏ hơn nhiều so với chênh lệch giữa hai giai đoạn |
| 10 | **LSTM / Transformer** | Khi hồi quy logistic còn thắng Random Forest trên test, thêm độ phức tạp chỉ làm tăng phương sai. Deep learning cần **nhiều** dữ liệu hơn, mà [§5.10.1](#5101-đường-cong-học--mô-hình-còn-thiếu-dữ-liệu) cho thấy dữ liệu hiện đang **thiếu** |
| 11 | **SHAP values** | Hữu ích để diễn giải từng dự đoán, nhưng không cải thiện hiệu năng; chỉ nên làm sau khi mô hình đã đạt ưu thế có ý nghĩa thống kê |

## 9.4. Hạ tầng kỹ thuật

- **Song song hoá huấn luyện.** Random Forest song song hoá hoàn toàn theo cây;
  hiện tại chạy tuần tự. Với 300 cây, tăng tốc gần tuyến tính theo số nhân CPU.
- **Tự động hoá quy trình đánh giá.** Gộp `makeEvidence.py` và hai script chạy chính
  thành một pipeline duy nhất sinh cả số liệu lẫn báo cáo, tránh việc tài liệu và mã
  nguồn nói hai chuyện khác nhau.
- **Ghi nhật ký thí nghiệm.** Mỗi lần chạy lưu lại cấu hình + kết quả + hash commit,
  để truy vết được con số nào đến từ phiên bản mã nào.

---

# 10. TÀI LIỆU THAM KHẢO

1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
   Learning* (2nd ed.). Springer — Chương 15: Random Forests.
3. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York
   Institute of Finance.
4. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley —
   Chương 7: Cross-Validation in Finance.
5. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal
   of Machine Learning Research, 12, 2825–2830.
6. Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised
   Classification Learning Algorithms*. Neural Computation, 10(7), 1895–1923 —
   cơ sở của kiểm định McNemar dùng ở [§5.6](#56-bốn-mã-một-bộ-siêu-tham-số--kiểm-định-thống-kê-mcnemar).

**Tài liệu nội bộ của đồ án:**

| Tài liệu | Nội dung |
| --- | --- |
| [`README.md`](README.md) | Tài liệu kỹ thuật đầy đủ: kiến trúc, quy trình, quy ước Git |
| [`docs/.reports/reportAlgorithm.md`](docs/.reports/reportAlgorithm.md) | Chuyên đề lý thuyết Random Forest — 12 mục, công thức và chứng minh |
| [`docs/.reports/reportResult.md`](docs/.reports/reportResult.md) | Báo cáo kết quả thực nghiệm chi tiết |
| [`docs/predictionGuide.md`](docs/predictionGuide.md) | Hướng dẫn dùng mô hình đã huấn luyện, bảng tra lỗi |
| [`docs/presentation/script.md`](docs/presentation/script.md) | Kịch bản thuyết trình 21 slide kèm số liệu minh chứng |

---

# 11. PHỤ LỤC

## Phụ lục A — Cấu trúc dự án

```text
BOT--Machine-Learning/
│
├── api/                          # TẦNG WEB — nằm NGOÀI src/, import một chiều
│   ├── main.py                   FastAPI: /api/models, /api/datasets, /api/upload, /api/stream
│   ├── streamService.py          Bọc replayEngine thành luồng SSE
│   ├── uploadService.py          Đọc csv/json/xlsx, ranh giới tin cậy
│   └── static/                   index.html + app.js + style.css, không cần build
│
├── config/                       # THAM SỐ THÍ NGHIỆM — dữ liệu, không phải mã
│   ├── base.json                 Phương pháp dùng chung: 27 đặc trưng + siêu tham số
│   ├── vnm/vcb/msn/mwg.json      Chỉ khai báo dữ liệu — kế thừa base.json
│   ├── regression.json           Nhánh B
│   └── adaptive.json             Tham số thay máu của rừng trượt
│
├── data/
│   ├── input/                    Dữ liệu thô (.csv / .xlsx)
│   └── output/                   Đồ thị, bảng kết quả
│
├── src/
│   ├── libraries/                # TẦNG LÕI — thuật toán thuần Python
│   │   ├── rfMath.py             Gini, Entropy, độ lợi, bootstrap, bộ tích luỹ quét ngưỡng
│   │   ├── decisionTree.py       Cây quyết định CART (phân loại + hồi quy)
│   │   ├── randomForest.py       Bagging + random feature + OOB + importance
│   │   ├── gradientBoosting.py   GB hồi quy và GB phân loại (bước Newton tại lá)
│   │   ├── linearModel.py        Hồi quy tuyến tính + logistic — mốc đối chứng
│   │   ├── slidingForest.py      Rừng biết thay máu theo thời gian
│   │   ├── modelStore.py         Tuần tự hoá cây và rừng ra JSON
│   │   └── rfPlot.py             14 hàm vẽ, không gắn với đề tài
│   │
│   ├── pipeline/                 # TẦNG BÀI TOÁN — chuỗi thời gian và phân tích kỹ thuật
│   │   ├── timePreprocess.py     Sắp xếp, khử trùng, điền tiến, tương quan
│   │   ├── technicalIndicators.py 20 chỉ báo, tất cả đều nhân quả
│   │   ├── featureBuilder.py     Thi hành BẢN ĐẶC TẢ đặc trưng từ config
│   │   ├── labeling.py           Nhãn xu hướng, mục tiêu hồi quy, cân bằng lớp
│   │   ├── splitter.py           Tách theo thời gian, walk-forward
│   │   ├── experiment.py         Điều phối: nạp → sạch → đặc trưng → nhãn → tách
│   │   ├── modelBundle.py        Gói mô hình = công thức + tham số + siêu dữ liệu
│   │   └── replayEngine.py       Bộ máy phát lại KHÔNG NHÌN TRƯỚC
│   │
│   ├── utilities/                # TIỆN ÍCH DÙNG CHUNG
│   │   ├── dataLoader.py         Đọc .csv/.xlsx, làm sạch theo dòng
│   │   ├── metrics.py            SSE, MSE, RMSE, MAE, MAPE, R², R²_adj, DirAcc
│   │   ├── metricsClassification.py Accuracy, Precision, Recall, F1, ROC-AUC, MCC
│   │   └── plotStyle.py          Bảng màu, kích thước, save_figure()
│   │
│   ├── notebooks/                01_eda · 02_features · 03_train_classify
│   │                             04_train_regress · 05_baseline_sklearn
│   ├── mainClassification.py     Chạy nhánh A đầu-cuối, có --save-model
│   ├── mainRegression.py         Chạy nhánh B đầu-cuối
│   └── mainPredict.py            Dự đoán từ gói mô hình, KHÔNG huấn luyện lại
│
├── models/                       Gói mô hình đã huấn luyện (.json)
├── docs/                         TOÀN BỘ TÀI LIỆU
├── tests/                        222 test tự động
│
├── Dockerfile                    Ảnh chỉ chứa phần chạy máy chủ
├── docker-compose.yml            Gắn models/ và data/ từ máy chủ nhà
├── .dockerignore                 Cắt ngữ cảnh build 710 MB → 4.7 MB
├── requirements.txt              Đầy đủ (phân tích + notebook + test)
├── requirements-api.txt          CHỈ 4 gói để chạy máy chủ
├── README.md                     Tài liệu kỹ thuật
└── REPORT.md                     Báo cáo này
```

## Phụ lục B — Danh sách đầy đủ 27 đặc trưng

| # | Tên | Chỉ báo gốc | Tham số | Nhóm |
| --- | --- | --- | --- | --- |
| 1 | `close_over_sma5` | `ratio_to_moving_average` | window 5 | Xu hướng |
| 2 | `close_over_sma10` | `ratio_to_moving_average` | window 10 | Xu hướng |
| 3 | `close_over_sma20` | `ratio_to_moving_average` | window 20 | Xu hướng |
| 4 | `close_over_sma50` | `ratio_to_moving_average` | window 50 | Xu hướng |
| 5 | `close_over_ema12` | `ratio_to_moving_average` | window 12, exp | Xu hướng |
| 6 | `sma5_above_sma20` | `crossover_indicator` | (5, 20) | Xu hướng |
| 7 | `macd_histogram` | `macd` → `ratio_series` | (12, 26, 9) | Xu hướng |
| 8 | `macd_over_signal` | `crossover_indicator` | — | Xu hướng |
| 9 | `rsi_7` | `relative_strength_index` | window 7 | Động lượng |
| 10 | `rsi_14` | `relative_strength_index` | window 14 | Động lượng |
| 11 | `roc_5` | `rate_of_change` | window 5 | Động lượng |
| 12 | `roc_10` | `rate_of_change` | window 10 | Động lượng |
| 13 | `stochastic_k` | `stochastic_oscillator` | window 14 | Động lượng |
| 14 | `stochastic_d` | `stochastic_oscillator` | smooth 3 | Động lượng |
| 15–19 | `return_lag_1..5` | `lag_series` | lag 1…5 | Động lượng |
| 20 | `bollinger_b` | `bollinger_bands` | (20, 2.0) | Biến động |
| 21 | `bollinger_width` | `bollinger_bands` | (20, 2.0) | Biến động |
| 22 | `atr_over_close` | `average_true_range` → `ratio_series` | window 14 | Biến động |
| 23 | `relative_range` | `relative_range` | — | Biến động |
| 24 | `volatility_10` | `rolling_standard_deviation` | window 10 | Biến động |
| 25 | `volatility_20` | `rolling_standard_deviation` | window 20 | Biến động |
| 26 | `volume_ratio_20` | `volume_ratio` | window 20 | Khối lượng |
| 27 | `obv_return` | `on_balance_volume` → `simple_returns` | — | Khối lượng |

Ngoài ra có 6 dãy trung gian (`returns`, `sma_5`, `sma_20`, `macd_*`, `atr_raw`,
`obv_raw`) được dựng với `keep: false` — dùng làm đầu vào cho các mục sau nhưng
không đưa vào tập đặc trưng cuối.

## Phụ lục C — Đồ thị đầu ra

**Sinh ra khi chạy hai script chính** (`data/output/`):

| File | Nội dung |
| --- | --- |
| `VNM_classification_class_distribution.png` | Phân bố lớp sau khi loại quan sát đứng yên |
| `VNM_classification_oob_curve.png` | Lỗi OOB theo số cây |
| `VNM_classification_feature_importance.png` | MDI đối chiếu permutation importance |
| `VNM_classification_confusion_matrix.png` | Ma trận nhầm lẫn |
| `VNM_classification_roc_curve.png` | Đường ROC kèm giá trị AUC |
| `VNM_classification_threshold_curve.png` | Accuracy / F1 / Balanced acc. theo ngưỡng |
| `VNM_classification_walk_forward.png` | Accuracy qua từng vòng kiểm định tiến dần |
| `VNM_regression_predicted_vs_actual.png` | Tán xạ dự đoán ↔ thực tế |
| `VNM_regression_series_comparison.png` | Chuỗi thực tế và dự đoán theo thời gian |
| `VNM_regression_residuals.png` | Phân bố sai số |
| `VNM_regression_oob_curve.png` | MSE ngoài túi theo số cây |
| `VNM_regression_walk_forward.png` | RMSE qua từng vòng kiểm định |

**Sinh ra khi chạy notebook:** `VNM_01_price_history.png`,
`VNM_01_class_distribution.png`, `VNM_02_indicators.png`, `VNM_02_correlation.png`,
`VNM_05_roc_comparison.png`.

**Ảnh minh chứng cho thuyết trình** (`docs/presentation/figures/`, sinh bởi
`makeEvidence.py` — 12 ảnh kèm `evidence.json` chứa toàn bộ số liệu gốc).

## Phụ lục D — Cách tái lập toàn bộ kết quả

```powershell
# 1. Chuẩn bị môi trường
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Kiểm chứng cài đặt trước khi tin vào bất kỳ con số nào
python -m pytest tests/ -q                        # → 222 passed

# 3. Chạy hai nhánh trên VNM
python src/mainClassification.py --config config/vnm.json --save-model models/vnm.json
python src/mainRegression.py     --config config/regression.json

# 4. Chạy ba mã còn lại với ĐÚNG bộ siêu tham số đó
python src/mainClassification.py --config config/vcb.json --save-model models/vcb.json
python src/mainClassification.py --config config/msn.json --save-model models/msn.json
python src/mainClassification.py --config config/mwg.json --save-model models/mwg.json

# 5. Sinh lại toàn bộ số liệu và ảnh minh chứng (~2,5 phút)
python docs/presentation/makeEvidence.py

# 6. Khởi động máy chủ dự đoán
python -m uvicorn api.main:application --port 8000
```

Mọi con số trong báo cáo này tái lập được với `random_state = 42`. Nếu kết quả khác
đi, kiểm tra trước hết: **nguồn dữ liệu**, **khoảng thời gian**, và giá trị
**`labeling.horizon`** trong file cấu hình.

---

<div align="center">

**BÁO CÁO ĐỒ ÁN — MINI PROJECT MACHINE LEARNING**

Dự đoán xu hướng giá cổ phiếu bằng Random Forest tự cài đặt

VNM · VCB · MSN · MWG · 2016–2026 · 222 test · `random_state = 42`

*"Không giấu con số xấu — đo nó."*

</div>
