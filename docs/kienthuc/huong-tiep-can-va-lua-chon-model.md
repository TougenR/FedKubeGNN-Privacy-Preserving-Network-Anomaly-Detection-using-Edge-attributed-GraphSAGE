# Hướng tiếp cận bài toán và lựa chọn mô hình

## 1. Bài toán cần giải quyết

Mục tiêu của hệ thống là phát hiện hành vi độc hại trong lưu lượng mạng IoT,
hướng tới môi trường Kubernetes. Dữ liệu đầu vào là các bản ghi network flow
trong IoT-23, không phải ảnh:

```text
IP nguồn, IP đích, protocol, port, duration,
số packet, số byte, trạng thái TCP, service, ...
```

Mỗi flow cần được phân loại thành một nhãn như `Benign`, `DDoS`, `C&C`,
`PartOfAHorizontalPortScan` hoặc một loại tấn công khác. Vì mỗi flow được biểu
diễn thành một cạnh của đồ thị, đây là bài toán **Network Intrusion Detection
kết hợp multi-class edge classification**.

## 2. Hướng tiếp cận của hệ thống

```text
+-----------------------------+
| IoT-23 conn.log.labeled     |
| Mỗi dòng = một network flow |
+--------------+--------------+
               |
               v
+-----------------------------+
| Tiền xử lý                  |
|                             |
| - Làm sạch missing value    |
| - Encode categorical        |
| - log1p dữ liệu lệch        |
| - StandardScaler            |
| - Chuẩn hóa label           |
+--------------+--------------+
               |
               v
+-----------------------------+
| Chuyển thành graph          |
|                             |
| Node = địa chỉ IP           |
| Edge = network flow         |
| Edge feature = thuộc tính   |
| Edge label = loại tấn công  |
+--------------+--------------+
               |
               v
+-----------------------------+
| E-GraphSAGE                 |
|                             |
| Tổng hợp hành vi flow       |
| xung quanh mỗi IP           |
+--------------+--------------+
               |
               v
+-----------------------------+
| Edge representation         |
|                             |
| [embedding IP nguồn         |
|  embedding IP đích          |
|  đặc trưng flow]            |
+--------------+--------------+
               |
               v
+-----------------------------+
| MLP classifier              |
+--------------+--------------+
               |
               v
+-----------------------------+
| Benign / DDoS / C&C / ...   |
+-----------------------------+
```

Ý tưởng quan trọng là model không chỉ nhìn từng flow riêng lẻ mà còn nhìn mối
quan hệ giữa flow đó với các flow khác trong mạng.

## 3. Vì sao cần ngữ cảnh giữa các flow?

Xét một flow đơn lẻ:

```text
10.0.0.5 -> 10.0.0.20
port = 80
duration = 0.1 s
packets = 3
```

Nếu chỉ nhìn flow này, nó có thể giống một kết nối HTTP bình thường. Tuy nhiên,
khi đặt flow trong topology đầy đủ:

```text
                    +--> 10.0.0.20:80
                    |
10.0.0.5 -----------+--> 10.0.0.21:80
                    |
                    +--> 10.0.0.22:80
                    |
                    +--> 10.0.0.23:80
                    |
                    +--> 10.0.0.24:80
```

Ta nhận ra `10.0.0.5` đang kết nối hàng loạt máy trong thời gian ngắn. Đây có
thể là horizontal port scan, botnet propagation, DDoS preparation hoặc malware
reconnaissance.

```text
Một flow riêng lẻ     -> có thể trông bình thường
Nhiều flow liên quan  -> hình thành mẫu hành vi tấn công
```

GNN được sử dụng để học những mẫu quan hệ này.

## 4. Vì sao không ưu tiên CNN thuần?

### 4.1 CNN giả định dữ liệu có cấu trúc lưới

CNN được thiết kế cho dữ liệu có cấu trúc đều đặn như ảnh, âm thanh hoặc chuỗi
thời gian. Ví dụ, mỗi pixel trong ảnh có các vị trí lân cận cố định và một
kernel có thể trượt trên toàn bộ ảnh:

```text
+---+---+---+
| x | x | x |
+---+---+---+
| x | x | x |   <--- kernel 3 x 3
+---+---+---+
| x | x | x |
+---+---+---+
```

CNN phù hợp khi:

- dữ liệu có thứ tự không gian hoặc thời gian cố định;
- các vị trí liền kề mang ý nghĩa;
- cùng một kernel có thể tái sử dụng ở mọi vị trí;
- việc dịch chuyển mẫu không làm thay đổi bản chất của mẫu.

### 4.2 Mạng máy tính là một cấu trúc bất quy tắc

Topology mạng không có khái niệm tự nhiên như trái, phải, trên hoặc dưới:

```text
             IP B
              |
              |
IP A -------- IP D -------- IP E
              |
              +------------ IP F
              |
             IP C
```

Mỗi IP có số lượng hàng xóm khác nhau. Nếu ép topology thành một ma trận để áp
dụng CNN, việc đổi thứ tự IP sẽ tạo ra một "ảnh" khác dù graph thật không thay
đổi:

```text
[A, B, C, D, E] != [D, A, E, C, B]
```

Vì vậy CNN 2D không phải lựa chọn tự nhiên cho topology mạng.

### 4.3 CNN 1D trên vector flow cũng có hạn chế

Có thể áp dụng CNN 1D lên một vector feature:

```text
[duration, bytes, packets, port, protocol, ...]
                    |
                  CNN 1D
                    |
                 classifier
```

Tuy nhiên, thứ tự các cột trong dữ liệu bảng thường không có ý nghĩa không
gian. Việc `duration` đứng cạnh `orig_bytes`, hoặc `protocol` đứng cạnh
`service`, không tạo ra cùng loại locality như các pixel cạnh nhau. Do đó phép
trượt và chia sẻ kernel của CNN không có cơ sở tự nhiên mạnh trên các cột flow.

Quan trọng hơn, nếu mỗi flow được đưa độc lập vào CNN thì model không biết các
flow khác có cùng IP nguồn hoặc IP đích:

```text
Flow 1 --> CNN --> dự đoán
Flow 2 --> CNN --> dự đoán
Flow 3 --> CNN --> dự đoán

Flow 1 không biết Flow 2 và Flow 3 tồn tại.
```

Với bài toán chỉ phân loại từng vector flow độc lập, MLP, Random Forest,
XGBoost hoặc LightGBM thường là các baseline tự nhiên hơn CNN.

## 5. GNN khai thác topology như thế nào?

GNN làm việc trực tiếp với node, edge và neighborhood. Mỗi node chỉ nhận thông
tin từ những node hoặc cạnh thực sự kết nối với nó:

```text
             Flow 1
      IP A ----------\
                      \
             Flow 2    v
      IP B ----------> IP D
                      ^
             Flow 3  /
      IP C ----------/
```

Tại IP D:

```text
Embedding mới của D
        =
Embedding cũ của D
        +
Thông tin tổng hợp từ Flow 1, Flow 2 và Flow 3
```

GNN vẫn có thể hoạt động khi số IP thay đổi, mỗi IP có bậc khác nhau, thứ tự
đánh số IP thay đổi hoặc xuất hiện IP mới chưa có trong tập train. Đây là một
lợi thế quan trọng của cách học inductive trong GraphSAGE.

## 6. Vì sao chọn E-GraphSAGE?

GraphSAGE thông thường chủ yếu truyền node feature:

```text
h_u --------------------------> node v
```

Trong dữ liệu network flow, node chỉ đại diện cho IP và không có feature hành
vi phong phú. Thông tin quan trọng nằm trên cạnh, chẳng hạn protocol, bytes,
packets, port và connection state. E-GraphSAGE đưa thông tin này vào message:

```text
h_u + protocol + bytes + packets + state
                         |
                         v
                       node v
```

Có thể tóm tắt sự khác biệt như sau:

```text
GraphSAGE:
    học ai kết nối với ai

E-GraphSAGE:
    học ai kết nối với ai
              +
    kết nối đó diễn ra như thế nào
```

Trong implementation hiện tại, mỗi message được tạo từ embedding node nguồn
và đặc trưng flow:

```text
m_(u->v) = ReLU(W_msg [h_u || e_uv] + b_msg)

a_v      = Mean({m_(u->v) : u thuộc N(v)})

h'_v     = ReLU(W_upd [h_v || a_v] + b_upd)
```

Sau hai layer, mỗi flow được biểu diễn bằng embedding của hai IP đầu cuối cùng
đặc trưng flow gốc:

```text
z_uv = [h_u || h_v || e_uv]
```

Vector này được đưa qua MLP để dự đoán nhãn của flow.

## 7. CNN và E-GraphSAGE nhìn thấy gì?

```text
CNN/MLP trên từng flow:

Flow 1 --> model --> dự đoán
Flow 2 --> model --> dự đoán
Flow 3 --> model --> dự đoán


E-GraphSAGE:

Flow 1 --+
         |
Flow 2 --+--> embedding hành vi của IP --> phân loại Flow 1
         |                              --> phân loại Flow 2
Flow 3 --+                              --> phân loại Flow 3
```

| Tiêu chí | CNN thuần | E-GraphSAGE |
|---|---|---|
| Cấu trúc dữ liệu tự nhiên | Lưới hoặc chuỗi | Graph |
| Đơn vị phân loại | Vector, ảnh hoặc đoạn chuỗi | Network flow |
| Khai thác topology IP | Khó hoặc không | Có |
| Liên hệ giữa nhiều flow | Thường không | Có |
| Số hàng xóm thay đổi | Không tự nhiên | Hỗ trợ tự nhiên |
| Thứ tự đánh số IP thay đổi | Có thể ảnh hưởng | Không nên ảnh hưởng |
| IP mới | Cần thiết kế thêm | Có tính inductive |
| Scan, DDoS và botnet phân tán | Hạn chế | Phù hợp hơn |
| Chi phí tính toán | Thường thấp hơn | Cao hơn |
| Độ phức tạp triển khai | Thấp hơn | Cao hơn |

## 8. Khi nào CNN vẫn phù hợp?

CNN vẫn là lựa chọn hợp lý nếu đầu vào có locality rõ ràng, ví dụ:

- chuỗi byte của packet;
- chuỗi packet theo thời gian;
- ma trận traffic theo các cửa sổ thời gian;
- spectrogram hoặc biểu diễn traffic dạng ảnh có ý nghĩa không gian.

Ví dụ:

```text
Packet payload:
[0x45, 0x00, 0x00, 0x54, ...]
                 |
               CNN 1D
```

Trong trường hợp này, các byte hoặc timestep liền kề có quan hệ tự nhiên, nên
phép convolution có cơ sở rõ ràng hơn.

## 9. Giới hạn và cách kiểm chứng lựa chọn

GNN không mặc định luôn tốt hơn CNN hoặc model tabular. Nếu nhãn của flow chỉ
phụ thuộc vào các feature cục bộ như duration, bytes, packets và protocol, còn
topology không bổ sung thông tin, XGBoost hoặc MLP có thể hiệu quả hơn và rẻ
hơn đáng kể.

Việc chọn E-GraphSAGE dựa trên giả thuyết:

> Hành vi tấn công không chỉ thể hiện trong từng flow mà còn thể hiện qua quan
> hệ và mẫu kết nối giữa nhiều host.

Giả thuyết này đặc biệt phù hợp với DDoS, botnet, horizontal port scan, C&C và
malware propagation. Tuy nhiên, kết quả thực nghiệm với các baseline mới là
bằng chứng cuối cùng cho thấy topology và edge-aware message passing thực sự
mang lại lợi ích trên IoT-23.

Repository hiện so sánh E-GraphSAGE với GCN, GraphSAGE, SAGE-Edge-Concat và
GAT. Khi trình bày kết quả, cần dùng macro-F1 và F1 theo từng lớp thay vì chỉ
dựa vào accuracy vì IoT-23 mất cân bằng lớp rất mạnh.

## 10. Tài liệu và mã nguồn liên quan

- Paper: [E-GraphSAGE: A Graph Neural Network based Intrusion Detection System
  for IoT](https://arxiv.org/abs/2103.16329).
- E-GraphSAGE layer và classifier: [`src/model.py`](../../src/model.py).
- Chuyển network flow thành graph: [`src/graph_build.py`](../../src/graph_build.py).
- Tiền xử lý IoT-23: [`src/preprocess.py`](../../src/preprocess.py).
- Huấn luyện và đánh giá: [`src/train.py`](../../src/train.py) và
  [`src/evaluate.py`](../../src/evaluate.py).
