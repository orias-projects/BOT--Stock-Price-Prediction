# Kịch bản thuyết trình — Dự đoán xu hướng giá cổ phiếu bằng Random Forest

**Thời lượng:** ~21 phút trình bày + hỏi đáp
**Số slide:** 21
**Ảnh minh chứng:** 12 ảnh tại `docs/presentation/figures/slide_{số}_{mô tả}.png`

> Mọi con số trong kịch bản này được sinh ra từ `docs/presentation/makeEvidence.py`.
> Chạy lại script là toàn bộ ảnh và số liệu tự cập nhật. Nếu sửa dữ liệu hoặc
> cấu hình, **chạy lại rồi cập nhật kịch bản** — đừng để slide và mã nguồn nói
> hai chuyện khác nhau.
>
> ```
> .venv/Scripts/python docs/presentation/makeEvidence.py          # tất cả (~2.5 phút)
> .venv/Scripts/python docs/presentation/makeEvidence.py 12 14    # chỉ vài slide
> ```

---

## Luận điểm xuyên suốt

Bài này **không** bán một mô hình dự đoán chứng khoán. Bài này chứng minh rằng
quy trình học máy đã được làm đúng, và chính vì làm đúng nên kết quả mới khiêm
tốn. Ba câu phải đọng lại trong đầu hội đồng:

1. **Quy trình sạch, và tôi chứng minh được điều đó** — bằng phép thử xáo nhãn,
   không phải bằng lời hứa.
2. **Accuracy 60% mà tôi báo cáo còn khó hơn 90% mà nhiều báo cáo khác đưa ra**
   — vì 90% kia đến từ rò rỉ dữ liệu, và tôi tái hiện được đúng con số đó.
3. **Tôi biết mô hình của mình yếu ở đâu, và có số liệu cho từng điểm yếu.**

Câu chốt để dùng nhiều lần: *"Em không giấu con số xấu, em đo nó."*

---

# PHẦN I — ĐẶT VẤN ĐỀ (slide 1–3, ~2 phút)

## Slide 1 — Trang bìa
**Thời lượng:** 20 giây · **Ảnh:** không

**Trên slide**
- Tiêu đề: *Dự đoán xu hướng giá cổ phiếu bằng Random Forest tự cài đặt*
- Phụ đề: *Một nghiên cứu về quy trình học máy trung thực trên dữ liệu tài chính*
- Tên, lớp, giảng viên hướng dẫn
- Góc dưới: `github.com/Arisu171/RF-Stock-Prediction` · 222 test · 4 mã cổ phiếu

**Lời thoại**
> Em xin chào thầy cô. Đề tài của em là dự đoán xu hướng tăng giảm giá cổ phiếu
> bằng Random Forest, trong đó thuật toán được cài đặt hoàn toàn bằng Python
> thuần, không dùng thư viện học máy.
>
> Em xin nói trước một điều để thầy cô biết mình sắp nghe gì: kết quả cao nhất
> trong bài này là 60%, và em sẽ dành phần lớn thời gian để giải thích **vì sao
> con số đó lại là kết quả tốt**.

**Chuyển:** *"Trước hết là bài toán."*

---

## Slide 2 — Bài toán và dữ liệu
**Thời lượng:** 50 giây · **Ảnh:** không (dùng bảng)

**Trên slide**

| | |
|---|---|
| Bài toán | Phân loại nhị phân — giá sau 5 phiên **tăng** hay **giảm** |
| Dữ liệu | 4 mã HOSE: VNM, VCB, MSN, MWG — cùng mốc thời gian |
| Quy mô | 2.456 mẫu/mã sau khi làm sạch · 27 đặc trưng kỹ thuật |
| Chia tập | Train 1.719 · Validate 368 · Test 369 · gap 5 phiên |
| Thuật toán | Random Forest tự cài đặt · đối chứng: hồi quy logistic tự cài đặt |

**Lời thoại**
> Em không dự đoán giá — em dự đoán **hướng**. Cụ thể: cho dữ liệu đến hết ngày
> hôm nay, sau năm phiên nữa giá đóng cửa cao hơn hay thấp hơn.
>
> Đây là bài toán phân loại nhị phân, nên có một mốc so sánh rất rõ ràng: tung
> đồng xu được 50%. Mọi con số em đưa ra sau đây, thầy cô cứ đối chiếu với 50%.
>
> Dữ liệu là bốn mã trên sàn HOSE, lấy cùng một mốc thời gian, mỗi mã khoảng
> 2.400 mẫu sau khi làm sạch.

**Chuyển:** *"Nhưng trước khi chạy bất kỳ thí nghiệm nào, em đã phải định nghĩa thế nào là thành công."*

---

## Slide 3 — Thế nào là thành công (đặt ra TRƯỚC khi chạy)
**Thời lượng:** 50 giây · **Ảnh:** không

**Trên slide** — sáu tiêu chí, chưa có kết quả:
1. Accuracy validate > 55% và vượt **mọi** đối chứng
2. ROC-AUC validate > 0.55
3. Chênh lệch train ↔ validate < 10%
4. Độ lệch chuẩn walk-forward < 5%
5. F1 của **mọi** lớp ≥ 0.40
6. Test không giảm quá 5% so với validate

> 🔒 Sáu tiêu chí này được ghi vào README **trước** khi chạy thí nghiệm đầu tiên.

**Lời thoại**
> Đây là phần em muốn nhấn mạnh nhất về mặt phương pháp. Sáu tiêu chí này được
> viết vào README **trước** khi em chạy thí nghiệm đầu tiên.
>
> Lý do rất đơn giản: nếu đặt tiêu chí sau khi đã thấy kết quả, thì tiêu chí nào
> cũng đạt — mình chỉ việc chọn ngưỡng vừa khít với con số mình có. Cách đó
> không chứng minh được gì cả.
>
> Cuối bài em sẽ quay lại đúng sáu dòng này và đối chiếu từng dòng, kể cả dòng
> không đạt.

**Chuyển:** *"Bây giờ em xin trình bày hệ thống."*

---

# PHẦN II — HỆ THỐNG (slide 4–7, ~3 phút)

## Slide 4 — Kiến trúc ba tầng
**Thời lượng:** 45 giây · **Ảnh:** không (sơ đồ vẽ trên slide)

**Trên slide** — sơ đồ ba khối xếp chồng, mũi tên chỉ một chiều đi xuống:

```
  api/          FastAPI · SSE · dashboard         ← biết về web
  ────────────────────────────────────────
  src/pipeline/ chuỗi thời gian · chỉ báo · nhãn  ← biết về chứng khoán
  ────────────────────────────────────────
  src/libraries/ cây · rừng · boosting · tuyến tính ← CHỈ biết toán
```

> Quy tắc: **tầng dưới không bao giờ import tầng trên.**
> `src/libraries/` không có một dòng nào nhắc tới "giá", "cổ phiếu" hay "VNM".

**Lời thoại**
> Hệ thống chia ba tầng, và ràng buộc quan trọng nhất là chiều phụ thuộc: tầng
> dưới không bao giờ biết tầng trên tồn tại.
>
> Cụ thể, trong `src/libraries` không có một dòng nào nhắc tới giá, cổ phiếu hay
> tên mã. Nó chỉ biết ma trận và nhãn. Nhờ vậy nếu mai em đổi sang bài toán
> khác, tầng đó dùng lại nguyên vẹn.
>
> Mọi giá trị cụ thể — tên mã, đường dẫn, siêu tham số — nằm trong file cấu hình
> JSON chứ không nằm trong mã nguồn.

**Chuyển:** *"Tầng dưới cùng là phần em tự viết."*

---

## Slide 5 — Random Forest tự cài đặt
**Thời lượng:** 60 giây · **Ảnh:** không (danh sách + snippet)

**Trên slide**
- Cây CART: Gini / Entropy, cắt theo ngưỡng, dừng theo độ sâu và số mẫu lá
- Bootstrap + chọn ngẫu nhiên đặc trưng tại mỗi nút (`sqrt`)
- Bỏ phiếu mềm theo xác suất, không phải bỏ phiếu cứng
- Lỗi Out-of-Bag — validate miễn phí, không tốn dữ liệu
- MDI + permutation importance

> **Tối ưu đáng kể nhất:** tìm ngưỡng cắt từ O(n²) xuống **O(n log n)** bằng một
> lượt quét tích luỹ → thời gian huấn luyện từ ~2 phút xuống **2,9 giây** (nhanh
> hơn 40 lần), kết quả giống hệt.

**Lời thoại**
> Toàn bộ thuật toán em tự cài đặt: cây CART, bootstrap, chọn ngẫu nhiên đặc
> trưng, bỏ phiếu mềm, lỗi out-of-bag, hai loại đo tầm quan trọng đặc trưng.
>
> Có một chi tiết kỹ thuật em muốn khoe. Bản đầu tiên huấn luyện mất khoảng hai
> phút vì mỗi lần thử một ngưỡng cắt là một lần quét lại toàn bộ mẫu — độ phức
> tạp bình phương. Em thay bằng một lượt quét duy nhất, vừa đi vừa cộng dồn số
> đếm mỗi lớp sang trái. Từ hai phút xuống 2,9 giây, nhanh hơn 40 lần, và kết
> quả giống hệt từng chữ số.
>
> Chi tiết này quan trọng vì nó cho phép em chạy hàng chục thí nghiệm mà thầy cô
> sắp thấy — nếu mỗi lần chạy mất hai phút thì em đã không đo được nhiều như vậy.

**Chuyển:** *"Đầu vào của rừng là 27 đặc trưng."*

---

## Slide 6 — 27 đặc trưng và hai cách đo tầm quan trọng
**Thời lượng:** 50 giây · **Ảnh:** `slide_06_dacTrung.png`

**Trên slide** — ảnh + hai dòng ghi chú:
- MDI đo cây **dùng** đặc trưng nhiều đến đâu (trên tập huấn luyện)
- Permutation đo mô hình **mất** bao nhiêu khi đặc trưng bị phá (trên validate)

**Lời thoại**
> 27 đặc trưng đều là chỉ báo kỹ thuật chuẩn: trung bình động, RSI, MACD,
> Bollinger, ATR, biến động, khối lượng, và một số độ trễ của lợi suất.
>
> Em đo tầm quan trọng bằng hai cách khác nhau, và chỗ đáng chú ý là **chúng
> không đồng ý với nhau**. Biểu đồ trái là MDI — cây dùng đặc trưng nào nhiều.
> Biểu đồ phải là permutation — phá đặc trưng đó đi thì mô hình mất bao nhiêu.
>
> Thầy cô nhìn `volatility_10`: nó đứng đầu ở cả hai, nên đây là đặc trưng thật.
> Nhưng `close_over_sma10` thì MDI xếp hạng sáu, còn permutation cho giá trị
> **âm** — nghĩa là phá nó đi mô hình còn tốt lên. Đó là đặc trưng cây bám vào
> nhưng không mang thông tin tổng quát.
>
> Em giữ nguyên cả 27 chứ không cắt bớt, vì việc cắt dựa trên tập validate lại
> là một dạng rò rỉ khác.

**Chuyển:** *"Nhắc đến rò rỉ — đây là phần em đầu tư nhiều công nhất."*

---

## Slide 7 — Bốn nguyên tắc chống rò rỉ
**Thời lượng:** 60 giây · **Ảnh:** `slide_07_chongRoRi.png`

**Trên slide** — ảnh sơ đồ trục thời gian + bốn nguyên tắc:
1. **Chỉ báo nhân quả** — mọi cửa sổ trượt chỉ đọc quá khứ, không đọc tương lai
2. **Nhãn dịch về sau** — nhãn của ngày *t* nằm ở *t+5*, nên tại *t* nó chưa tồn tại
3. **Tách theo thời gian** — không xáo trộn, train luôn nằm trước validate
4. **gap = horizon = 5** — chặn mẫu cuối train chạm vào mẫu đầu validate

**Lời thoại**
> Rò rỉ dữ liệu là khi mô hình vô tình nhìn thấy tương lai trong lúc huấn luyện.
> Trên dữ liệu chuỗi thời gian, chuyện này xảy ra dễ đến mức nó là lỗi phổ biến
> nhất của cả lĩnh vực.
>
> Em chặn bằng bốn lớp. Ba lớp đầu khá hiển nhiên. Lớp thứ tư — cái khoảng trống
> năm phiên giữa các tập — thì tinh vi hơn, em xin giải thích.
>
> Nhãn của em nhìn về tương lai năm phiên. Nếu tập huấn luyện kết thúc đúng ngày
> tập validate bắt đầu, thì năm mẫu cuối cùng của tập huấn luyện mang nhãn được
> tính từ giá **nằm trong** tập validate. Chỉ năm mẫu thôi, nhưng đó vẫn là rò
> rỉ. Nên em bỏ đúng năm phiên ở mỗi ranh giới.
>
> Bốn nguyên tắc này **làm kết quả của em xấu đi**. Đó là lý do em tin chúng
> đúng.

**Chuyển:** *"Nhưng nói thì ai cũng nói được. Phần tiếp theo em chứng minh."*

---

# PHẦN III — CHỨNG MINH QUY TRÌNH ĐÚNG (slide 8–10, ~4 phút) ⭐

> **Đây là phần quan trọng nhất của bài. Nếu thiếu thời gian, cắt phần II chứ
> đừng cắt phần này.**

## Slide 8 — Ba kiểm chứng độc lập
**Thời lượng:** 70 giây · **Ảnh:** `slide_08_kiemChungCaiDat.png`

**Trên slide** — ảnh + ba ô kết quả:

| Kiểm chứng | Kết quả |
|---|---|
| Bootstrap hội tụ về 1/e | đo **0.3673** · lý thuyết **0.36788** |
| Đối chiếu scikit-learn | Accuracy 0.6087 vs 0.6141 · AUC 0.6476 vs 0.6446 · OOB 0.3973 vs 0.4043 |
| Bộ kiểm thử | **222 test**, tất cả xanh |

**Lời thoại**
> Câu hỏi đầu tiên thầy cô có quyền hỏi: tự cài đặt thì lấy gì đảm bảo đúng?
> Em trả lời bằng ba cách độc lập.
>
> Thứ nhất, một tính chất toán học có đáp án biết trước. Khi lấy mẫu bootstrap n
> lần từ n mẫu, tỷ lệ mẫu không được chọn hội tụ về 1 trên e. Em đo được 0,3673,
> lý thuyết là 0,36788. Khớp đến ba chữ số. Đây là kiểm chứng em thích nhất vì
> nó không thể ăn may.
>
> Thứ hai, em chạy scikit-learn với đúng cấu hình đó trên đúng dữ liệu đó. Cả ba
> chỉ số chênh nhau **dưới 1%**. Và bốn đặc trưng quan trọng nhất được hai bên
> xếp hạng **giống hệt thứ tự**.
>
> Thứ ba là 222 test tự động cho từng thành phần.
>
> Em xin nói rõ: em dùng scikit-learn để **kiểm chứng**, không dùng để chạy.
> Toàn bộ kết quả trong bài đến từ mã em viết.

**Chuyển:** *"Cài đặt đúng rồi. Nhưng quy trình có sạch không? Câu này khó hơn."*

---

## Slide 9 — Phép thử xáo nhãn ⭐
**Thời lượng:** 80 giây · **Ảnh:** `slide_09_xaoNhan.png`

**Trên slide** — ảnh + hai kết luận:

| | Nhãn thật | Nhãn xáo ngẫu nhiên |
|---|---|---|
| Train accuracy | 0.6597 | 0.6719 · 0.6632 · 0.6812 |
| **ROC-AUC validate** | **0.6418** | **0.4134 · 0.4535 · 0.5445** |

> ① Quy trình **không** rò rỉ · ② Train accuracy là con số **vô nghĩa**

**Lời thoại**
> Đây là thí nghiệm em tâm đắc nhất. Em xáo ngẫu nhiên toàn bộ nhãn huấn luyện —
> phá sạch mọi mối liên hệ giữa đặc trưng và nhãn — rồi huấn luyện lại y hệt.
>
> Nếu quy trình của em có rò rỉ, mô hình vẫn sẽ tìm được cách gian lận và AUC
> trên validate vẫn cao. Kết quả: AUC sụp xuống 0,41 — 0,45 — 0,54, tức là quanh
> mức ngẫu nhiên, có lần còn dưới. Không có đường tắt nào tồn tại trong dữ liệu.
> **Quy trình sạch.**
>
> Nhưng thí nghiệm này còn trả lời một câu thứ hai, và câu này mới thú vị. Thầy
> cô nhìn cột train accuracy: nhãn thật cho 0,6597, nhãn **ngẫu nhiên hoàn toàn**
> cho 0,67 — 0,66 — 0,68. Cao hơn.
>
> Nghĩa là train accuracy 66% của em **không chứa một chút thông tin nào** về
> chất lượng mô hình. Nó chỉ đo khả năng học thuộc. Vì vậy trong toàn bộ bài này,
> em không dùng train accuracy để kết luận bất cứ điều gì.
>
> Em nói điều này vì em thấy nhiều báo cáo lấy chính con số đó làm kết quả chính.

**Chuyển:** *"Và nếu bỏ mấy nguyên tắc kia đi thì được bao nhiêu? Em đã thử."*

---

## Slide 10 — Vì sao báo cáo khác đạt 90% ⭐⭐ (slide quan trọng nhất)
**Thời lượng:** 90 giây · **Ảnh:** `slide_10_roRiDuLieu.png`

**Trên slide** — ảnh + bảng:

| Cách làm | Accuracy | |
|---|---|---|
| Tách theo **thời gian** (dự án dùng) | **0.5292** | ✅ trung thực |
| Xáo trộn ngẫu nhiên rồi tách 85/15 | 0.5718 | ⚠️ +4,3 điểm miễn phí |
| Xáo trộn + để cây mọc tự do | 0.6612 | ⚠️ +13,2 điểm |
| Báo cáo trên chính tập huấn luyện | **1.0000** | ❌ 100% |

> **Cùng dữ liệu. Cùng thuật toán. Chỉ đổi cách tách tập.**

**Lời thoại**
> Slide này trả lời trước câu hỏi mà em nghĩ thầy cô sẽ hỏi: *"Em ơi, sao
> accuracy của em thấp thế, mấy bạn khác báo 90%?"*
>
> Em xin trả lời bằng thí nghiệm thay vì lý lẽ. Bốn cột này dùng **cùng một bộ
> dữ liệu, cùng một thuật toán, cùng một đoạn mã**. Thứ duy nhất thay đổi là cách
> tách tập.
>
> Cột một, cách của em: 0,529.
>
> Cột hai, em xáo trộn ngẫu nhiên rồi mới tách — đúng như `train_test_split` mặc
> định mà rất nhiều bài dùng. 0,572. Tăng hơn bốn điểm phần trăm, chỉ vì bây giờ
> mẫu ngày mai và mẫu hôm nay cùng nằm trong tập huấn luyện, mà hai mẫu liền kề
> thì gần như giống nhau. Mô hình không dự đoán — nó nhận ra hàng xóm.
>
> Cột ba, thêm việc để cây mọc tự do không giới hạn độ sâu: 0,661. Đã hơn 13 điểm.
>
> Cột bốn, báo cáo trên chính tập huấn luyện: **100%**. Tuyệt đối. Và hoàn toàn
> vô giá trị.
>
> Cho nên: con số 0,529 của em và con số 0,90 của một báo cáo khác không so sánh
> được với nhau, vì chúng đo hai thứ khác nhau. Em tin con số của mình vì em biết
> chính xác mình đã bỏ đi những gì để có nó.

**Chuyển:** *"Với quy trình đó, kết quả thật là như thế nào."*

---

# PHẦN IV — KẾT QUẢ (slide 11–15, ~5,3 phút)

## Slide 11 — Bốn mã, một bộ siêu tham số duy nhất
**Thời lượng:** 70 giây · **Ảnh:** `slide_11_bonMa.png`

**Trên slide** — ảnh + bảng:

| Mã | Valid acc | p (McNemar) | Test acc | p (McNemar) |
|---|---|---|---|---|
| VNM *(dò tham số)* | 0.6087 | 0.289 | 0.5237 | 0.805 |
| VCB | 0.5359 | 0.089 | 0.4840 | 0.063 |
| MSN | 0.5669 | 0.208 | 0.5219 | 0.242 |
| MWG | 0.4955 | **0.042** ⚠️ | 0.5318 | 1.000 |

> Siêu tham số dò **chỉ trên VNM**, ba mã còn lại không được chạm vào.

**Lời thoại**
> Em huấn luyện bốn mô hình riêng cho bốn mã, nhưng dùng **một bộ siêu tham số
> duy nhất**, và bộ đó chỉ được dò trên VNM. Ba mã còn lại em không đụng vào tham
> số một lần nào — chúng là kiểm chứng sạch.
>
> VNM đạt 0,6087 trên validate. Nhưng em không dừng ở con số đó, vì câu hỏi thật
> sự là: chênh lệch so với đoán bừa có phân biệt được với may rủi không?
>
> Em dùng kiểm định McNemar. Với VNM, p bằng 0,289 — **không** đạt mức ý nghĩa
> 5%. Nghĩa là em không thể bác bỏ giả thuyết "chênh lệch này là ngẫu nhiên".
>
> Và có một kết quả em phải báo cáo dù nó không đẹp: MWG trên validate cho p bằng
> 0,042, tức là **có ý nghĩa thống kê** — nhưng theo chiều **ngược lại**. Mô hình
> tệ hơn đoán bừa một cách có ý nghĩa. Em để nguyên trong báo cáo.
>
> Tóm lại: không mã nào có ưu thế dương đạt ý nghĩa thống kê. Đây là kết luận
> trung thực của bài.

**Chuyển:** *"Vậy Random Forest có hơn được một mô hình đơn giản không? Em phải kiểm tra."*

---

## Slide 12 — Đối chứng hồi quy logistic ⭐
**Thời lượng:** 80 giây · **Ảnh:** `slide_12_doiChungHoiQuy.png`

**Trên slide** — ảnh + bảng:

| | VNM | VCB | MSN | MWG | |
|---|---|---|---|---|---|
| **Validate** — Rừng | **0.6087** | **0.5359** | **0.5669** | 0.4955 | rừng thắng **3/4** |
| **Validate** — Hồi quy | 0.5353 | 0.5135 | 0.4739 | **0.5290** | |
| **Test** — Rừng | 0.5237 | 0.4840 | 0.5219 | **0.5318** | rừng thắng **1/4** |
| **Test** — Hồi quy | **0.5822** | **0.5228** | **0.5473** | 0.4864 | |

> Thứ hạng **đảo chiều** giữa hai tập: 3/4 → 1/4

**Lời thoại**
> Một mô hình phức tạp chỉ đáng dùng nếu nó thắng được mô hình đơn giản nhất.
> Nên em cài thêm hồi quy logistic — cũng bằng Python thuần, cũng có kiểm thử —
> và cho nó ăn **đúng 27 đặc trưng đó**.
>
> Trên tập validate, Random Forest thắng ba trên bốn mã. Nghe có vẻ ổn.
>
> Nhưng trên tập test, kết quả **đảo ngược**: hồi quy logistic thắng ba trên bốn.
> VNM chẳng hạn — rừng 0,524, hồi quy 0,582.
>
> Em xin nhấn mạnh vì đây là phát hiện đáng giá nhất của cả bài: không phải "mô
> hình tuyến tính tốt hơn". Kết luận đúng là **ngay cả việc chọn mô hình nào tốt
> hơn cũng không ổn định qua các giai đoạn**. Nếu em chỉ nhìn tập validate, em đã
> kết luận Random Forest thắng và tự tin sai.
>
> Đây chính là lý do em giữ tập test và chỉ mở nó ra một lần duy nhất ở cuối.

**Chuyển:** *"Và nếu đổi hẳn cách đặt bài toán thì sao?"*

---

## Slide 13 — Nếu dự đoán ĐỘ LỚN thay vì hướng: SSE và R² ⭐
**Thời lượng:** 65 giây · **Ảnh:** `slide_13_hoiQuyGiaTri.png`

**Trên slide** — ảnh + bảng R² trên tập validate:

| Mã | Random Forest | Hồi quy tuyến tính | Đoán bằng trung bình |
|---|---|---|---|
| VNM | **+0.072** | −0.018 | −0.015 |
| VCB | −0.042 | −0.080 | −0.015 |
| MSN | −0.039 | −0.053 | −0.017 |
| MWG | −0.010 | −0.020 | −0.000 |

> VNM: SST = 0.20266 · SSE = 0.18815 · RMSE = 0.02228 · MAE = 0.01677
> **R² = 1 − SSE/SST** — R² < 0 nghĩa là SSE > SST, tức **tệ hơn đoán bằng trung bình**

**Lời thoại**
> Ở slide 2 em nói em dự đoán hướng chứ không dự đoán giá. Slide này là lý do,
> và em đo chứ không nói suông.
>
> Em giữ nguyên toàn bộ 27 đặc trưng, chỉ đổi biến mục tiêu: thay vì nhãn tăng
> giảm, giờ mô hình phải dự đoán **tỷ suất sinh lợi sau năm phiên** — một con số
> thực. Rồi em đo bằng đúng bộ chỉ số hồi quy trong `metrics.py`: SST, SSE, RMSE,
> MAE và R².
>
> Kết quả ở cột R². Chỉ VNM dương, và cũng chỉ 0,072 — nghĩa là mô hình giải
> thích được **7,2% biến động**, còn 92,8% là nhiễu.
>
> Ba mã còn lại **R² âm**. Em xin nói rõ ý nghĩa của con số âm, vì nó không phải
> đơn thuần là "kém". R² bằng một trừ SSE chia SST. R² âm tức là **SSE lớn hơn
> SST** — sai số của mô hình lớn hơn sai số của việc đoán bừa bằng trung bình tập
> huấn luyện. Cột xám trong biểu đồ chính là mô hình không học gì cả, và nó
> **thắng** ở ba trên bốn mã.
>
> Biểu đồ bên phải cho thấy vì sao. Trục ngang là tỷ suất thực tế, trục dọc là
> dự đoán. Nếu dự đoán tốt, các điểm phải nằm trên đường chéo đỏ. Thực tế chúng
> nằm bẹp thành một dải ngang quanh số không: biên độ dự đoán chỉ 0,039 trong khi
> thực tế trải 0,158 — **hẹp hơn hơn bốn lần**. Mô hình đã học được điều đúng đắn
> nhất mà nó có thể học, là *"đừng đoán mạnh"*.
>
> Đây chính là lý do đề tài đóng khung thành bài toán phân loại. Dự đoán **độ
> lớn** của biến động giá là bài toán mà dữ liệu này không giải được. Dự đoán
> **hướng** thì còn có một chút tín hiệu — và "một chút" đó là toàn bộ những gì
> em báo cáo ở các slide trước.

**Chuyển:** *"Thêm một góc nữa: mô hình có bền theo thời gian không?"*

---

## Slide 14 — Mô hình có hạn sử dụng
**Thời lượng:** 50 giây · **Ảnh:** `slide_14_thoaiHoa.png`

**Trên slide** — ảnh + dòng số:

> **0.6259 → 0.6463 → 0.5850 → 0.5102 → 0.5168**
> 737 phiên được chấm, chia đều năm giai đoạn liên tiếp sau khi huấn luyện xong

**Lời thoại**
> Em phát lại toàn bộ dữ liệu sau tập huấn luyện, từng phiên một, đúng như mô
> hình sẽ chạy ngoài thực tế: mỗi ngày dự đoán một lần, năm phiên sau mới biết
> đúng hay sai.
>
> Chia 737 phiên đó thành năm giai đoạn liên tiếp, accuracy đi từ 0,63 xuống
> 0,52. Hai giai đoạn cuối gần như bằng đoán bừa.
>
> Nghĩa là mô hình **có hạn sử dụng**. Nếu triển khai thật, phải huấn luyện lại
> định kỳ — và biểu đồ này cho biết chu kỳ đó khoảng bao lâu. Đây là thông tin mà
> một con số accuracy duy nhất không bao giờ nói ra được.

**Chuyển:** *"Bây giờ em xin quay lại sáu tiêu chí ban đầu."*

---

## Slide 15 — Đối chiếu sáu tiêu chí
**Thời lượng:** 60 giây · **Ảnh:** `slide_15_tieuChi.png`

**Trên slide** — ảnh (bảng ✔/✘) — đạt **5/6**

**Lời thoại**
> Đây là sáu dòng ở slide 3, giờ có kết quả.
>
> Năm tiêu chí đạt. Tiêu chí thứ sáu **không đạt**: accuracy trên test giảm 8,5
> điểm so với validate, trong khi em đặt ngưỡng 5. Em để nguyên dấu đỏ.
>
> Và em xin tự phê bình tiêu chí thứ nhất. Về mặt con số nó đạt — 0,6087 cao hơn
> cả đoán bừa lẫn hồi quy logistic. Nhưng McNemar cho p bằng 0,289, nghĩa là
> chênh lệch đó **không phân biệt được với may rủi**. Em ghi thẳng điều đó lên
> slide, vì nếu chỉ để dấu tick xanh thì em đang mắc đúng cái lỗi mà em phê phán
> ở slide 10.
>
> Đánh giá trung thực: 5 trên 6 tiêu chí về mặt hình thức, còn về mặt thực chất
> thì mô hình chưa chứng minh được ưu thế đáng tin cậy so với đoán bừa.

**Chuyển:** *"Phần cuối, em xin demo hệ thống chạy thật."*

---

# PHẦN V — DEMO TRỰC TIẾP (slide 16–17, ~2,5 phút) ⭐

> **Chuẩn bị trước khi thuyết trình:**
> ```
> .venv/Scripts/python -m uvicorn api.main:application --port 8000
> ```
> Mở sẵn `http://localhost:8000`, chọn VNM, **chưa bấm Start**.
> Chụp sẵn ảnh dự phòng vào `figures/slide_16_dashboard.png` và
> `figures/slide_17_upload.png` — phòng khi máy chiếu hoặc mạng trục trặc.

## Slide 16 — Dashboard phát lại theo luồng
**Thời lượng:** 100 giây · **Ảnh:** demo trực tiếp (dự phòng: `slide_16_dashboard.png`)

**Kịch bản thao tác**
1. Chọn mã VNM → bấm **Start** → biểu đồ nến chạy
2. Chỉ vào **ba đường accuracy**: tĩnh · thích nghi · đoán bừa (đường đoán bừa để mờ)
3. Chỉ vào mũi tên dự đoán trên nến — dự đoán hiện **trước**, kết quả đến **sau 5 phiên**
4. Bấm **Pause**, chỉ vào một dự đoán sai

**Lời thoại**
> Đây là hệ thống chạy thật, phát lại dữ liệu theo luồng qua Server-Sent Events.
> Không dùng thư viện biểu đồ nào — em vẽ trực tiếp bằng Canvas 2D.
>
> Thầy cô để ý ba điểm.
>
> Thứ nhất, phần dữ liệu dùng để huấn luyện **bị bỏ qua** — luồng bắt đầu từ sau
> đó. Cho mô hình chạy lại trên dữ liệu nó đã học thuộc thì con số vô nghĩa.
>
> Thứ hai, ba đường accuracy: mô hình tĩnh, mô hình thích nghi, và đoán bừa. Em
> để đường đoán bừa mờ đi nhưng vẫn hiện — vì nếu giấu nó đi thì hai đường kia
> trông ấn tượng hơn thực tế nhiều.
>
> Thứ ba, và đây là điểm em thiết kế kỹ nhất: dự đoán hiện lên ngay tại phiên
> đang chạy, nhưng chỉ được **chấm điểm sau năm phiên**, khi kết quả thật xuất
> hiện. Hệ thống giữ một hàng đợi chờ. Nếu chấm ngay thì đó là nhìn trộm tương
> lai — đúng cái lỗi ở slide 10, chỉ khác là lần này nó xảy ra lúc chạy thật.
>
> Tốc độ: một dự đoán trọn vẹn — nhận 60 dòng thô, dựng đủ 27 chỉ báo, chạy qua
> 300 cây — mất **1,7 mili giây**. Phát lại toàn bộ 763 phiên hết 1,2 giây.
> Thị trường Việt Nam mỗi ngày một phiên — nên về tốc độ thì thừa sức chạy thời
> gian thực. Vấn đề không nằm ở tốc độ, mà ở độ chính xác.

**Chuyển:** *"Tab thứ hai cho phép nạp dữ liệu ngoài."*

---

## Slide 17 — Tab tải lên
**Thời lượng:** 50 giây · **Ảnh:** demo trực tiếp (dự phòng: `slide_17_upload.png`)

**Kịch bản thao tác**
1. Sang tab **Upload** → kéo thả file CSV
2. Chọn mô hình → **Start** → chạy đúng như tab trước
3. (Nếu còn thời gian) thả file sai cột để cho xem thông báo lỗi

**Lời thoại**
> Tab này nhận file CSV, JSON hoặc XLSX, và chạy qua **đúng cùng một bộ máy phát
> lại** — em không viết một đường xử lý riêng, vì hai đường song song thì sớm
> muộn cũng lệch nhau.
>
> Điều em muốn chỉ ra là phần kiểm tra đầu vào. File phải có đủ cột và tối thiểu
> **50 dòng** — vì chỉ báo dài nhất của em cần 50 phiên khởi động. Nếu thiếu cột,
> hệ thống báo rõ cột nào thiếu và file đang có cột nào, chứ không ném ra một lỗi
> Python khó hiểu.

**Chuyển:** *"Cuối cùng là những gì em rút ra."*

---

# PHẦN VI — BÀI HỌC VÀ GIỚI HẠN (slide 18–21, ~3,3 phút)

## Slide 18 — Bài học 1: mô hình còn thiếu dữ liệu
**Thời lượng:** 45 giây · **Ảnh:** `slide_18_duongCongHoc.png`

**Trên slide** — ảnh + dòng số:

| Số mẫu | 343 | 687 | 1.031 | 1.375 | 1.719 |
|---|---|---|---|---|---|
| ROC-AUC | 0.4496 | 0.4985 | 0.5052 | 0.6339 | **0.6418** |

**Lời thoại**
> Câu hỏi tự nhiên: 60% là do thiếu dữ liệu, hay do dữ liệu vốn không chứa nhiều
> tín hiệu hơn?
>
> Em huấn luyện lại với 20%, 40%, 60%, 80% và 100% dữ liệu. Đường cong vẫn đang
> **dốc lên** ở điểm cuối — chưa chạm trần. Nghĩa là thêm dữ liệu vẫn còn giúp
> được, và giới hạn hiện tại một phần đến từ việc em chỉ có khoảng 1.700 mẫu
> huấn luyện.
>
> Đây là hướng cải thiện rõ ràng nhất, và nó không đòi hỏi thuật toán phức tạp
> hơn — chỉ cần nhiều lịch sử hơn.

**Chuyển:** *"Bài học thứ hai đi ngược trực giác."*

---

## Slide 19 — Bài học 2: mô hình đang thừa tham số gấp 10 lần
**Thời lượng:** 45 giây · **Ảnh:** `slide_19_kichThuoc.png`

**Trên slide** — ảnh + bảng:

| Cấu hình | Tham số | AUC validate | Acc test |
|---|---|---|---|
| 300 cây, sâu 5 *(đang dùng)* | 19.670 | 0.6476 | 0.5237 |
| 100 cây, sâu 4 | 4.135 | **0.6493** | 0.5404 |
| **50 cây, sâu 4** | **1.955** | 0.6418 | **0.5460** |
| 30 cây, sâu 3 | 735 | 0.6212 | 0.5292 |

**Lời thoại**
> Em quét thử nhiều kích thước mô hình. Bản đang dùng có gần 20.000 tham số.
>
> Bản nhỏ hơn **mười lần** — chưa tới 2.000 tham số — cho AUC gần như y hệt, và
> accuracy trên test còn **cao hơn**: 0,546 so với 0,524.
>
> Nói cách khác, 90% số tham số hiện tại đang dùng để học thuộc nhiễu chứ không
> học quy luật. Nếu em có thời gian làm lại, em sẽ chọn bản 1.955 tham số.
>
> Em giữ nguyên bản lớn trong báo cáo vì đó là bản em đã dò tham số từ đầu, và
> đổi sang bản nhỏ **sau khi nhìn thấy điểm test** thì lại là một dạng gian lận
> khác.

**Chuyển:** *"Bài học thứ ba là một thí nghiệm không thành công."*

---

## Slide 20 — Bài học 3: cho mô hình tự học thêm không giúp gì
**Thời lượng:** 50 giây · **Ảnh:** `slide_20_thichNghi.png`

**Trên slide** — ảnh + bảng:

| Cách làm | Accuracy | Thời gian | Số cây |
|---|---|---|---|
| **Cây tĩnh** (không đụng gì) | **0.5767** | **1,1 s** | 300 |
| Thay cây, dữ liệu gần đây (400 mẫu) | 0.5658 | 1,7 s | 300 |
| Thay cây, đủ dữ liệu như cây cũ | 0.5672 | 4,3 s | 300 |
| Chỉ thêm cây, không xoá | 0.5712 | 4,2 s | 390 |
| *Đoán bừa* | *0.5522* | — | — |

**Lời thoại**
> Ý tưởng ban đầu của em là cho rừng tự cập nhật khi dữ liệu mới chảy vào: định
> kỳ trồng cây mới trên dữ liệu gần đây và loại bớt cây cũ.
>
> Em thử bốn cách, kể cả cách công bằng nhất — cây mới được học trên **đúng
> lượng dữ liệu** bằng cây cũ, để phép so sánh không lẫn hai biến.
>
> Cả bốn đều **thua** cây tĩnh, và tốn gấp bốn lần thời gian. Càng thay đổi nhiều
> thì càng kém.
>
> Em giải thích thế này: dữ liệu gần đây ít hơn và nhiễu hơn, nên cây mới yếu hơn
> cây cũ. Việc thay cây chỉ có lợi khi thị trường thật sự đổi chế độ, mà trong
> khoảng thời gian em chọn thì điều đó không xảy ra — em cố ý chọn bốn mã không
> có biến cố cấu trúc lớn.
>
> Em giữ thí nghiệm này trong báo cáo vì một kết quả âm được đo cẩn thận vẫn là
> một kết quả.

**Chuyển:** *"Slide cuối cùng."*

---

## Slide 21 — Giới hạn và kết luận
**Thời lượng:** 60 giây · **Ảnh:** không

**Trên slide** — hai cột:

**Đã làm được**
- Quy trình học máy đầy đủ, chống rò rỉ bốn lớp, chứng minh bằng phép thử xáo nhãn
- Thuật toán tự cài đặt, kiểm chứng ba cách độc lập, 222 test
- Bốn mã, một bộ siêu tham số, có đối chứng tuyến tính và kiểm định thống kê
- Hệ thống phát lại thời gian thực, đóng gói Docker

**Chưa làm được / giới hạn**
- Ưu thế so với đoán bừa **chưa đạt ý nghĩa thống kê** ở bất kỳ mã nào
- Mô hình thoái hoá rõ từ giai đoạn thứ ba (~300 phiên) → phải huấn luyện lại định kỳ
- Chưa tính phí giao dịch, trượt giá, thanh khoản → **không dùng để đầu tư**
- Dữ liệu ~2.400 mẫu/mã, đường cong học cho thấy vẫn còn thiếu

**Lời thoại**
> Em xin kết lại bằng đúng những gì đo được.
>
> Về quy trình: em tin là đã làm đúng, và quan trọng hơn — em **chứng minh được**
> là đúng, bằng phép thử xáo nhãn và ba kiểm chứng cài đặt.
>
> Về kết quả: mô hình tốt nhất đạt 0,6087 trên validate. Nhưng ưu thế so với đoán
> bừa chưa đạt ý nghĩa thống kê ở bất kỳ mã nào, và trên tập test thì hồi quy
> logistic còn thắng ở ba trên bốn mã.
>
> Vì vậy kết luận trung thực của em là: **hệ thống này chứng minh được quy trình,
> chưa chứng minh được lợi nhuận.** Em không đề xuất dùng nó để đầu tư.
>
> Và em nghĩ đó mới là kết quả đúng. Nếu một dự án sinh viên chạy trong vài tuần
> mà dự đoán được thị trường chứng khoán, thì nhiều khả năng là quy trình có chỗ
> sai chứ không phải thị trường có chỗ hở.
>
> Em xin hết. Em cảm ơn thầy cô.

---

# CHUẨN BỊ CHO PHẦN HỎI ĐÁP

## Câu hỏi gần như chắc chắn được hỏi

**"Sao accuracy thấp thế? Mấy bạn khác 90%."**
> Dạ, em có chuẩn bị riêng cho câu này ở slide 10. *(quay lại slide 10)* Cùng
> dữ liệu, cùng thuật toán, chỉ đổi cách tách tập thì em cũng ra 0,66, và nếu
> báo cáo trên tập huấn luyện thì ra 100%. Em có thể ra con số đó trong năm phút.
> Em chọn không làm vậy.

**"Vậy dự án này có dùng được không?"**
> Dùng được để chứng minh quy trình, không dùng được để đầu tư. Em nói rõ ở
> slide 21. Giá trị của bài nằm ở chỗ em đo được **chính xác nó yếu ở đâu** —
> thoái hoá sau bao lâu, thiếu bao nhiêu dữ liệu, thừa bao nhiêu tham số.

**"Sao không dùng scikit-learn cho nhanh?"**
> Đề bài yêu cầu hiểu thuật toán. Và việc tự cài cho em thứ mà gọi thư viện
> không cho: khi em thấy tỷ lệ out-of-bag khớp 1/e đến ba chữ số, em biết chắc
> phần bootstrap của mình đúng. Em vẫn dùng scikit-learn — để kiểm chứng, ở
> slide 8.

**"Sao không dùng LSTM / Transformer / deep learning?"**
> Slide 12 trả lời được câu này. Hồi quy logistic — mô hình đơn giản nhất có thể
> — còn thắng Random Forest trên tập test. Khi mô hình tuyến tính chưa thua rõ
> ràng thì thêm độ phức tạp chỉ làm tăng phương sai. Đường cong học ở slide 18
> cho thấy vấn đề là **thiếu dữ liệu**, mà deep learning cần nhiều dữ liệu hơn
> chứ không phải ít hơn.

**"Kiểm định McNemar là gì?"**
> Nó chỉ nhìn những mẫu mà hai mô hình **bất đồng**, vì phần hai mô hình giống
> nhau không phân biệt được chúng. Nếu chênh lệch chỉ là may rủi thì số lần A
> đúng-B sai sẽ xấp xỉ số lần ngược lại. p-value là xác suất thấy chênh lệch
> lớn như quan sát được nếu thật ra hai mô hình ngang nhau.

**"Tại sao horizon 5 phiên mà không phải 1?"**
> Với horizon 1, phần lớn biến động là nhiễu vi cấu trúc và tỷ lệ phiên đi ngang
> rất cao. 5 phiên là một tuần giao dịch, đủ để tín hiệu kỹ thuật thể hiện.

**"Tại sao chọn đúng bốn mã đó?"**
> Em chấm điểm mức lệch của giai đoạn gần đây so với trước đó cho 12 mã, rồi
> chọn bốn mã ổn định nhất — VNM, VCB, MSN, MWG. FPT chẳng hạn bị loại vì điểm
> lệch 0,341, VIC 1,119. Nếu để mã có biến cố cấu trúc vào thì em không tách được
> "mô hình dở" khỏi "thị trường đổi chế độ".

## Câu hỏi khó — nếu bị hỏi

**"Em dò tham số trên VNM rồi báo cáo VNM, thế có thiên lệch không?"**
> Có, và em ghi rõ điều đó trong file cấu hình `config/vnm.json` cũng như trên
> slide 11. Chính vì vậy em thêm ba mã không được chạm vào tham số. Kết quả của
> ba mã đó thấp hơn VNM — đúng như dự đoán nếu có thiên lệch chọn mô hình. Em
> giữ VNM lại chính là để cho thấy độ lớn của thiên lệch đó.

**"Em nói tiêu chí 1 đạt nhưng McNemar không có ý nghĩa — vậy rốt cuộc là đạt hay không?"**
> Đạt theo đúng câu chữ em viết ra từ đầu, và **không** đạt theo tiêu chuẩn chặt
> hơn. Em không sửa lại tiêu chí sau khi thấy kết quả, mà ghi cả hai lên slide.
> Nếu làm lại, em sẽ viết tiêu chí 1 thành "vượt đối chứng với p < 0,05".

**"Sao không xáo trộn dữ liệu để có nhiều dữ liệu huấn luyện hơn?"**
> Đó chính xác là cột hai ở slide 10. Nó cho thêm 4,3 điểm accuracy, nhưng số đó
> không tồn tại ngoài thực tế — ngoài thực tế không ai có mẫu của ngày mai để
> huấn luyện.

---

# CHECKLIST TRƯỚC KHI THUYẾT TRÌNH

- [ ] `.venv/Scripts/python docs/presentation/makeEvidence.py` chạy xong, **12 ảnh** mới
- [ ] Số trong kịch bản khớp `figures/evidence.json`
- [ ] `.venv/Scripts/python -m pytest -q` → 222 passed
- [ ] Máy chủ chạy: `uvicorn api.main:application --port 8000`, mở sẵn trang, chọn sẵn VNM
- [ ] Ảnh dự phòng `slide_16_dashboard.png`, `slide_17_upload.png` đã chụp
- [ ] Slide 10, 12 và 13 đánh dấu sẵn để nhảy nhanh khi hỏi đáp
- [ ] Bấm thử Start/Pause một lần trên máy chiếu thật

# PHÂN BỔ THỜI GIAN

Cột "Phút" là tổng thời lượng thật của từng slide trong kịch bản, không phải
con số ước lượng — nên nếu tổng vượt quá thời gian được cấp, phải cắt thật chứ
không nói nhanh hơn.

| Phần | Slide | Phút | Cắt được? |
|---|---|---|---|
| I — Đặt vấn đề | 1–3 | 2,0 | slide 2 rút còn 20s |
| II — Hệ thống | 4–7 | 3,6 | ✂️ cắt slide 5, gộp 6 vào 7 |
| **III — Chứng minh** | **8–10** | **4,0** | ❌ **không cắt** |
| IV — Kết quả | 11–15 | 5,5 | slide 14 rút còn 30s |
| **V — Demo** | **16–17** | **2,5** | ❌ **không cắt slide 16** |
| VI — Bài học | 18–21 | 3,3 | ✂️ gộp 19 và 20 vào một slide |
| | | **20,9** | |

**Bản rút gọn 15 phút** — nếu hội đồng cấp ít thời gian, bỏ slide 5, 6, 14, 19,
20 và rút slide 2 xuống 20 giây. Còn lại 15 slide, ~14,5 phút, giữ nguyên toàn bộ
phần III, slide 12, slide 13 và demo.
