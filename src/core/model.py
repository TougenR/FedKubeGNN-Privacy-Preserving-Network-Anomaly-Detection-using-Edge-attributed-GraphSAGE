"""
model.py — E-GraphSAGE (chính) + 4 baseline đối chứng cho EDGE classification.

Mục tiêu: chứng minh việc dùng đặc trưng cạnh (E-GraphSAGE) tốt hơn các
cách không tận dụng đầy đủ đặc trưng cạnh. Tất cả 5 model cùng bài toán
edge classification, cùng head, chỉ khác phần sinh embedding node.

1. E-GraphSAGE (Task 1.9, độ khó ★★★★★)
   `EGraphSAGELayer` tự viết kế thừa `torch_geometric.nn.MessagePassing`.
   Trong `message()` GHÉP đặc trưng node NGUỒN + đặc trưng cạnh trước khi
   aggregate; trong `update()` ghép embedding gốc của node với thông điệp
   đã gộp. Đã có test ở `scripts/test_egraphsage_layer.py` và end-to-end
   ở `scripts/test_egraphsage_model.py`.

2. Baseline GCN (`GCNBaseline`)
   `GCNConv` có sẵn, KHÔNG nhận edge_attr. Lan truyền thuần node feature,
   mù hoàn toàn về đặc trưng cạnh. Đây chính là điểm yếu cố ý để đối
   chứng.

3. Baseline GraphSAGE (`GraphSAGEBaseline`)
   `SAGEConv` có sẵn, cũng KHÔNG nhận edge_attr. Tương tự GCNBaseline.

4. Baseline Phương án B (`SAGEEdgeConcatBaseline`)
   "Nhồi" đặc trưng cạnh vào node TRƯỚC khi vào SAGEConv: với mỗi node,
   mean đặc trưng các cạnh nối tới nó thành 1 vector, gắn vào node
   feature. Đây là bản "gần đúng" E-GraphSAGE mà không sửa `message()`.
   Thường kém hơn E-GraphSAGE vì mất locality + mất per-edge.

5. Baseline GAT (Graph Attention, `GATBaseline`)
   Dùng `GATv2Conv` có sẵn của PyG VỚI tham số ``edge_dim``. Đây là
   "cách thứ ba" để tận dụng edge feature: cơ chế attention khi tính
   trọng số message sẽ tham chiếu ``edge_attr`` của từng cạnh. Nhiều
   head song song (mặc định 4) cho đủ sức biểu diễn; layer cuối
   ``concat=False`` để gom chiều về ``hidden_dim`` rồi đưa vào head.
   Khác E-GraphSAGE: thay vì CONCAT edge vào message, GAT dùng edge để
   ĐIỀU CHỈNH TRỌNG SỐ attention trên message node→node.

Quy ước so sánh: cả 5 model dùng CHUNG head
`concat[h_u, h_v, edge_attr_goc] → Linear → ReLU → Dropout → Linear → num_classes`
— chỉ khác phần sinh embedding node. Đảm bảo so sánh công bằng (mọi
khác biệt đến từ cách xử lý edge feature trong message passing).

Lưu ý hạ tầng: mọi layer phải đặt trên `device` do train.py truyền vào;
không hardcode `.cuda()`.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv, MessagePassing, SAGEConv
from torch_geometric.utils import scatter


class EGraphSAGELayer(MessagePassing):
    """
    Một layer E-GraphSAGE (Edge-enhanced GraphSAGE).

    Ý tưởng cốt lõi
    ---------------
    Trong bài toán edge classification trên đồ thị (node = IP, cạnh = flow),
    thông tin phân biệt hành vi nằm ở **cạnh** (duration, bytes, cờ TCP bắt
    tay, port đích, …), không ở node. GraphSAGE gốc chỉ tổng hợp embedding
    từ các node láng giềng và BỎ QUA `edge_attr`. E-GraphSAGE khắc phục bằng
    cách **ghép đặc trưng cạnh vào message ngay trước khi aggregate**, để
    node đích nhận thông điệp đã được làm giàu bằng ngữ cảnh cạnh.

    Cơ chế từng bước
    ----------------
    Với `flow='source_to_target'` (mặc định của `MessagePassing`):

    1.  Với mỗi cạnh `(u, v)`, `x_j` là embedding của node **nguồn** `u`
        (PyG tự động lấy theo `edge_index[0]`). PHẢI dùng `x_j` (nguồn),
        không dùng `x_i` (đích), vì thông điệp mô tả "nguồn gửi gì đến đích".
    2.  `message(x_j, edge_attr)` ghép `m = concat([x_j, edge_attr])` rồi
        đưa qua `Linear(in_dim + edge_dim → out_dim) + ReLU`. Đây là chỗ
        E-GraphSAGE khác GraphSAGE gốc.
    3.  `aggregate` (mặc định `mean`) gộp các thông điệp tại mỗi node đích.
    4.  `update(aggr_out, x)` ghép `concat([x, aggr_out])` rồi đưa qua
        `Linear(in_dim + out_dim → out_dim) + ReLU`. PyG tự cung cấp `x`
        gốc (chưa qua aggregation) cho `update`.

    Args
    ----
    in_dim  : int — số chiều embedding node đầu vào.
    out_dim : int — số chiều embedding node đầu ra (sau layer này).
    edge_dim: int — số chiều đặc trưng cạnh (`edge_attr`).
    aggr    : str — kiểu aggregation; mặc định `'mean'` (đúng tinh thần
              GraphSAGE). Có thể đổi sang `'sum'` nếu cần.

    Ghi chú
    -------
    - ReLU đặt trong `message()` và `update()` để non-linearity xuất hiện
      sớm; nếu thấy mạng khó hội tụ có thể đẩy ra ngoài và dùng `ELU`/
      `PReLU` ở cấp model. Hiện tại giữ ReLU cho đơn giản và đúng spec.
    - Device-agnostic: không `.cuda()` ở đây; layer được dời lên `device`
      bởi `train.py` qua `model.to(device)`.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int,
                 aggr: str = 'mean'):
        super().__init__(aggr=aggr)

        # Message: concat [x_j (in_dim), edge_attr (edge_dim)] -> project + ReLU.
        self.lin_msg = nn.Linear(in_dim + edge_dim, out_dim)

        # Update: concat [x gốc (in_dim), aggr_out (out_dim)] -> project + ReLU.
        self.lin_upd = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x         : [N, in_dim]   — embedding node hiện tại.
        edge_index: [2, E]        — (nguồn, đích) của từng cạnh.
        edge_attr : [E, edge_dim] — đặc trưng cạnh.

        Returns
        -------
        [N, out_dim] — embedding mới sau layer.
        """
        # PyG tự tách `x` thành `x_j` (= x[edge_index[0]]) cho message(),
        # đồng thời cung cấp `x` gốc cho update(). Xem scripts/test_egraphsage_layer.py.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Thông điệp trên mỗi cạnh.

        Parameters
        ----------
        x_j       : [E, in_dim]   — embedding node NGUỒN của từng cạnh.
        edge_attr : [E, edge_dim] — đặc trưng cạnh.

        Returns
        -------
        [E, out_dim] — thông điệp đã được làm giàu và đưa về out_dim.
        """
        m = torch.cat([x_j, edge_attr], dim=-1)   # [E, in_dim + edge_dim]
        return torch.relu(self.lin_msg(m))        # [E, out_dim]

    def update(self, aggr_out: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        """
        Kết hợp embedding cũ của node với thông điệp đã aggregate.

        Parameters
        ----------
        aggr_out : [N, out_dim] — mean (hoặc sum) của incoming messages.
        x        : [N, in_dim]  — embedding gốc TRƯỚC khi vào layer này
                                  (PyG tự re-pass từ propagate()).

        Returns
        -------
        [N, out_dim] — embedding mới của node sau layer.
        """
        h = torch.cat([x, aggr_out], dim=-1)      # [N, in_dim + out_dim]
        return torch.relu(self.lin_upd(h))        # [N, out_dim]



class EGraphSAGE(torch.nn.Module):
    """
    Mô hình E-GraphSAGE hoàn chỉnh cho **edge classification** trên đồ thị
    IP-flow (IoT-23).

    Bài toán ở đây là **PHÂN LOẠI TỪNG CẠNH** (mỗi flow = 1 cạnh) — không phải
    node classification. Theo tinh thần E-GraphSAGE, đặc trưng cạnh là nguồn
    tín hiệu chính; node chỉ mang vector hằng (do ``graph_build`` sinh ra
    ``Data(x=ones([N, node_in_dim]))``).

    Kiến trúc (đã chốt trong CLAUDE.md mục 7):

        x ──► [EGraphSAGELayer × num_layers + Dropout] ──► h  (node embedding)
                                                             │
        edge_index: [2, E] gốc                               ▼
        edge_attr : [E, F]   gốc   ──► concat[h[src], h[dst], edge_attr] ──►
                                                              head (MLP)
                                                              │ Linear→ReLU→Drop→Linear
                                                              ▼
                                                    logits [E, num_classes]

    Phân biệt rõ hai tập cạnh:

        • ``edge_index_mp`` / ``edge_attr_mp`` (kích thước 2E) — dùng để
          MESSAGE PASSING vì đồ thị cần 2 chiều để lan truyền thông tin cả
          orig→resp lẫn resp→orig.
        • ``edge_index`` / ``edge_attr`` (kích thước E, cạnh gốc) — dùng để
          CLASSIFICATION: lấy embedding 2 đầu + đặc trưng gốc, đưa qua head.

    Args
    ----
    edge_dim     : int — số chiều đặc trưng cạnh (``Data.feature_dim``).
    num_classes  : int — số lớp đầu ra (``Data.num_classes``).
    node_in_dim  : int — số chiều vector node đầu vào (mặc định 1, tinh thần
                  E-GraphSAGE). Nên đặt bằng ``Data.x.shape[1]``.
    hidden_dim   : int — số chiều embedding ẩn (đọc từ ``config.yaml``).
    num_layers   : int — số layer E-GraphSAGE liên tiếp (≥ 1).
    dropout      : float — xác suất dropout, áp dụng giữa các layer và
                  bên trong head.

    Ghi chú
    -------
    - Device-agnostic: model không gọi ``.cuda()``. ``train.py`` sẽ chịu
      trách nhiệm ``model.to(device)`` và ``data.to(device)``.
    - Seed reproducible: trọng số khởi tạo phụ thuộc global seed; đặt
      ``torch.manual_seed(...)`` ngay đầu chương trình.
    - ``edge_dim`` / ``num_classes`` / ``node_in_dim`` ĐỌC ĐỘNG từ
      ``Data.feature_dim`` / ``Data.num_classes`` / ``Data.x.shape[1]`` —
      tuyệt đối KHÔNG hardcode.
    """

    def __init__(self, edge_dim: int, num_classes: int,
                 node_in_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"EGraphSAGE: num_layers={num_layers} < 1.")
        if hidden_dim < 1:
            raise ValueError(f"EGraphSAGE: hidden_dim={hidden_dim} < 1.")
        if edge_dim < 1:
            raise ValueError(f"EGraphSAGE: edge_dim={edge_dim} < 1.")
        if num_classes < 1:
            raise ValueError(f"EGraphSAGE: num_classes={num_classes} < 1.")

        # Metadata (đọc được từ state_dict để tái dựng model).
        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        # ----- Message passing stack -----
        # Layer 0: node_in_dim -> hidden_dim.
        # Layer 1..n-1: hidden_dim -> hidden_dim.
        self.layers = nn.ModuleList()
        self.layers.append(
            EGraphSAGELayer(in_dim=node_in_dim, out_dim=hidden_dim,
                            edge_dim=edge_dim, aggr='mean')
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                EGraphSAGELayer(in_dim=hidden_dim, out_dim=hidden_dim,
                                edge_dim=edge_dim, aggr='mean')
            )
        # Dropout xen giữa các layer (áp sau mỗi layer, kể cả layer cuối).
        self.dropout_between = nn.Dropout(p=dropout)

        # ----- Edge classification head -----
        # Vào: concat[h_u, h_v, e]  ->  hidden_dim + hidden_dim + edge_dim.
        # MLP: Linear -> ReLU -> Dropout -> Linear -> num_classes.
        head_in_dim = hidden_dim * 2 + edge_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Khởi tạo trọng số tuyến tính theo He (ReLU-friendly) cho ổn định
        # trên đồ thị sâu; default của PyTorch (kaiming_uniform_) cũng OK,
        # nhưng gọi tường minh để reproducible hoàn toàn khi đổi seed.
        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo trọng số Linear theo kaiming_uniform_ + bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_nodes(self, data) -> torch.Tensor:
        """Encode graph nodes once so routed heads can share the encoder."""
        h = data.x
        for layer in self.layers:
            h = layer(h, data.edge_index_mp, data.edge_attr_mp)
            h = self.dropout_between(h)
        return h

    def edge_representations(self, data, node_embeddings) -> torch.Tensor:
        """Build the exact edge representation consumed by every FedPer head."""
        src = data.edge_index[0]
        dst = data.edge_index[1]
        return torch.cat(
            [node_embeddings[src], node_embeddings[dst], data.edge_attr], dim=-1
        )

    def forward(self, data) -> torch.Tensor:
        """
        Chạy forward cho edge classification.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Bắt buộc có:
                - ``x``           : [N, node_in_dim]
                - ``edge_index_mp``: [2, 2E]   — cạnh 2 chiều (gốc + đảo).
                - ``edge_attr_mp`` : [2E, F]    — đặc trưng cho 2E cạnh MP.
                - ``edge_index``   : [2, E]     — cạnh GỐC.
                - ``edge_attr``    : [E, F]     — đặc trưng cạnh GỐC.

        Returns
        -------
        torch.Tensor
            ``logits`` shape ``[E, num_classes]`` — mỗi dòng là phân phối
            qua ``num_classes`` lớp cho flow tương ứng (cạnh gốc).
        """
        node_embeddings = self.encode_nodes(data)
        edge_repr = self.edge_representations(data, node_embeddings)
        return self.head(edge_repr)


class GCNBaseline(torch.nn.Module):
    """
    Baseline GCN cho **edge classification**.

    Điểm yếu cố ý (để đối chứng với E-GraphSAGE)
    ----------------------------------------------
    ``GCNConv(x, edge_index)`` chỉ nhận node feature + cấu trúc đồ thị,
    KHÔNG truyền được edge feature. Tức là trong LAN TRUYỀN, đặc trưng cạnh
    hoàn toàn **bị bỏ qua** — cũng giống GraphSAGE gốc. Đây chính là khoảng
    cách ta muốn chứng minh E-GraphSAGE cải thiện được.

    Vẫn dùng head GIỐNG E-GraphSAGE (concat[h_u, h_v, edge_attr_goc]) nên
    một phần tín hiệu cạnh vẫn được khai thác ở đầu ra. Sự khác biệt hiệu
    năng đến từ chỗ: embedding `h` của node đã mất edge_attr khi đi qua
    GCN layer, nên `h_u`, `h_v` kém giàu thông tin hơn.

    Args
    ----
    Cùng chữ ký với ``EGraphSAGE``: ``edge_dim``, ``num_classes``,
    ``node_in_dim``, ``hidden_dim``, ``num_layers``, ``dropout``.

    Ghi chú
    -------
    - Cùng seed + cùng head + cùng hidden_dim/num_layers/dropout với
      E-GraphSAGE ⇒ khác biệt chỉ đến từ phần message passing.
    - Không dùng edge_attr trong MP.
    """

    def __init__(self, edge_dim: int, num_classes: int,
                 node_in_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"GCNBaseline: num_layers={num_layers} < 1.")
        if hidden_dim < 1:
            raise ValueError(f"GCNBaseline: hidden_dim={hidden_dim} < 1.")
        if edge_dim < 1:
            raise ValueError(f"GCNBaseline: edge_dim={edge_dim} < 1.")
        if num_classes < 1:
            raise ValueError(f"GCNBaseline: num_classes={num_classes} < 1.")

        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        # GCN stack — KHÔNG truyền edge_attr vào message.
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout_between = nn.Dropout(p=dropout)

        # CÙNG head như E-GraphSAGE → so sánh công bằng.
        head_in_dim = hidden_dim * 2 + edge_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        x = data.x                                    # [N, node_in_dim]
        edge_index_mp = data.edge_index_mp            # [2, 2E] — KHÔNG edge_attr

        # === MESSAGE PASSING (không dùng edge_attr) ===
        h = x
        for layer in self.layers:
            h = layer(h, edge_index_mp)               # GCNConv: chỉ x + edge_index
            h = self.dropout_between(h)

        # === EDGE CLASSIFICATION (cùng head như E-GraphSAGE) ===
        edge_index = data.edge_index                  # [2, E]
        edge_attr = data.edge_attr                    # [E, F]
        src = edge_index[0]
        dst = edge_index[1]
        h_u = h[src]
        h_v = h[dst]
        edge_repr = torch.cat([h_u, h_v, edge_attr], dim=-1)  # [E, 2H+F]
        logits = self.head(edge_repr)                          # [E, num_classes]
        return logits


class GraphSAGEBaseline(torch.nn.Module):
    """
    Baseline GraphSAGE cho **edge classification**.

    Cùng triết lý so sánh với ``GCNBaseline``: dùng ``SAGEConv`` có sẵn —
    KHÔNG nhận edge feature trong lan truyền. Trong đồ thị IoT-23, hành vi
    phân biệt nằm trên cạnh nên cả GCN và SAGE gốc đều mù với tín hiệu đó
    trong khi E-GraphSAGE thì không.

    Args / ghi chú: y hệt ``GCNBaseline`` (chỉ khác lớp conv bên trong).
    """

    def __init__(self, edge_dim: int, num_classes: int,
                 node_in_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"GraphSAGEBaseline: num_layers={num_layers} < 1.")
        if hidden_dim < 1:
            raise ValueError(f"GraphSAGEBaseline: hidden_dim={hidden_dim} < 1.")
        if edge_dim < 1:
            raise ValueError(f"GraphSAGEBaseline: edge_dim={edge_dim} < 1.")
        if num_classes < 1:
            raise ValueError(f"GraphSAGEBaseline: num_classes={num_classes} < 1.")

        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        # SAGEConv stack — KHÔNG truyền edge_attr.
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout_between = nn.Dropout(p=dropout)

        # CÙNG head như E-GraphSAGE.
        head_in_dim = hidden_dim * 2 + edge_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index_mp = data.edge_index_mp            # [2, 2E] — KHÔNG edge_attr

        h = x
        for layer in self.layers:
            h = layer(h, edge_index_mp)               # SAGEConv: không dùng edge_attr
            h = self.dropout_between(h)

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        src = edge_index[0]
        dst = edge_index[1]
        h_u = h[src]
        h_v = h[dst]
        edge_repr = torch.cat([h_u, h_v, edge_attr], dim=-1)
        logits = self.head(edge_repr)
        return logits


class SAGEEdgeConcatBaseline(torch.nn.Module):
    """
    Phương án B (fallback) — "nhồi" edge feature vào node TRƯỚC khi lan truyền.

    Ý tưởng
    -------
    Thay vì sửa ``message()`` để ghép edge_attr vào từng message (cách
    E-GraphSAGE làm), ở đây ta MEAN-toàn-bộ-đặc-trưng-cạnh-nối-tới-mỗi-node
    thành 1 vector, ghép vào node feature ngay đầu vào, rồi chạy ``SAGEConv``
    bình thường trên node feature mới. Đây chính là "Phương án B" trong
    CLAUDE.md mục 7 — bản "gần đúng" của E-GraphSAGE.

    Vì sao thường YẾU hơn E-GraphSAGE (lập luận để đưa vào báo cáo)
    ----------------------------------------------------------------
    1.  **Mất locality:** mỗi node biết đặc trưng của TẤT CẢ cạnh nối tới
        nó ngay ở layer 0; sau khi lan truyền 1 hop, thông tin đó có thể
        đến nút xa hơn 2-hop. E-GraphSAGE giữ locality chặt: mỗi message
        chỉ mang edge_attr của CẠNH ĐÓ.
    2.  **Mất per-edge:** nhiều cạnh khác nhau cùng incident tới 1 node
        bị mean thành 1 vector duy nhất — không phân biệt được. E-GraphSAGE
        mỗi cạnh có message riêng.

    Mục đích trong thực nghiệm: chứng minh "chỉ ghép edge vào node input là
    chưa đủ — phải can thiệp ở mức message mới tận dụng hết edge feature
    trong lan truyền".

    Args
    ----
    edge_dim, num_classes, node_in_dim, hidden_dim, num_layers, dropout:
        cùng chữ ký với các model khác.
    agg : {'mean', 'sum'}
        Cách tổng hợp đặc trưng cạnh cho mỗi node. Mặc định ``'mean'``
        (chuẩn hơn ``'sum'`` vì 1 node có thể có rất nhiều cạnh nối tới
        trong IoT-23).

    Quy ước tổng hợp
    ----------------
    Với mỗi node ``v``, ta tổng hợp đặc trưng của các cạnh trong tập
    ``edge_index_mp`` (2E cạnh, gốc + đảo) mà ``v`` là **target**
    (``edge_index_mp[1] == v``). Vì tập cạnh MP đã có cả 2 chiều, cách
    này thu được TẤT CẢ cạnh incident tới ``v`` trong đồ thị vô hướng
    tương ứng — mỗi cạnh gốc ``(u, v)`` đóng góp attr một lần cho v
    (qua chiều thuận) và một lần cho u (qua chiều nghịch). Mỗi attr
    cạnh được dùng đúng **2 lần** trong tổng hợp (đúng cấu trúc đồ thị
    vô hướng).
    """

    def __init__(self, edge_dim: int, num_classes: int,
                 node_in_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5,
                 agg: str = 'mean'):
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                f"SAGEEdgeConcatBaseline: num_layers={num_layers} < 1."
            )
        if hidden_dim < 1:
            raise ValueError(
                f"SAGEEdgeConcatBaseline: hidden_dim={hidden_dim} < 1."
            )
        if edge_dim < 1:
            raise ValueError(
                f"SAGEEdgeConcatBaseline: edge_dim={edge_dim} < 1."
            )
        if num_classes < 1:
            raise ValueError(
                f"SAGEEdgeConcatBaseline: num_classes={num_classes} < 1."
            )
        if agg not in ('mean', 'sum'):
            raise ValueError(
                f"SAGEEdgeConcatBaseline: agg='{agg}' không hỗ trợ "
                f"(chỉ 'mean' hoặc 'sum')."
            )

        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.agg = agg

        # Sau khi "nhồi" edge_attr vào node, dimension = node_in_dim + edge_dim.
        node_concat_dim = node_in_dim + edge_dim

        # SAGEConv stack vẫn không nhận edge_attr (chỉ khác input node dim).
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(node_concat_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout_between = nn.Dropout(p=dropout)

        # CÙNG head như E-GraphSAGE.
        head_in_dim = hidden_dim * 2 + edge_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        x = data.x                                    # [N, node_in_dim]
        edge_index_mp = data.edge_index_mp            # [2, 2E]
        edge_attr_mp = data.edge_attr_mp              # [2E, F]
        N = x.size(0)

        # === Bước riêng của baseline này: tổng hợp edge_attr cho mỗi node ===
        # Tổng hợp trên edge_index_mp (đã gồm gốc + đảo); mỗi attr cạnh vô
        # hướng được count đúng 2 lần (một cho mỗi đầu).
        node_edge_agg = scatter(
            edge_attr_mp,                  # [2E, F]
            edge_index_mp[1],              # target
            dim=0,
            dim_size=N,
            reduce=self.agg,                # 'mean' | 'sum'
        )                                  # [N, F]

        # Nhồi vào node feature.
        h = torch.cat([x, node_edge_agg], dim=-1)  # [N, node_in_dim + F]

        # === SAGEConv stack (giống GraphSAGEBaseline nhưng input giàu hơn) ===
        for layer in self.layers:
            h = layer(h, edge_index_mp)
            h = self.dropout_between(h)

        # === EDGE CLASSIFICATION (cùng head) ===
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        src = edge_index[0]
        dst = edge_index[1]
        h_u = h[src]
        h_v = h[dst]
        edge_repr = torch.cat([h_u, h_v, edge_attr], dim=-1)
        logits = self.head(edge_repr)
        return logits


class GATBaseline(torch.nn.Module):
    """
    Baseline GAT v2 cho **edge classification** — dùng GATv2Conv có sẵn.

    Đặc điểm nổi bật so với các baseline khác
    ------------------------------------------
    GATv2Conv(..., edge_dim=F) chấp nhận đặc trưng cạnh và DÙNG NÓ
    khi tính attention score (không giống GCN/SAGE gốc — bỏ qua edge).
    Cú pháp:

        α_{ij} = softmax_j(LeakyReLU(aᵀ [W·h_i || W·e_{ij} || W·h_j]))

    Vì vậy edge_attr được đưa vào QUYẾT ĐỊNH BAO NHIÊU THÔNG TIN từ
    node j truyền sang node i, theo từng cạnh (per-edge, không gộp như
    SAGEEdgeConcatBaseline). Vẫn khác E-GraphSAGE ở chỗ: E-GraphSAGE
    CONCAT edge vào message trước khi đưa qua Linear; GAT dùng edge để
    RE-WEIGHT message gốc (không trộn thông tin).

    Kiến trúc (đã chốt)
    -------------------
    Stack GATv2Conv với chiến lược chiều:

        Layer 0   : GATv2Conv(node_in_dim, hidden_dim, heads=heads,
                                  concat=True, edge_dim=F) ⇒ output
                                  [N, hidden_dim * heads].
        Layer 1..num_layers-2 (giữa): cùng cấu hình nhưng
                      in_channels = hidden_dim * heads.
        Layer cuối (num_layers ≥ 2) : cùng cấu hình nhưng
                      concat=False ⇒ output [N, hidden_dim]
                      (mean qua các head).
        num_layers=1                : 1 layer với concat=False để ra
                      hidden_dim trực tiếp (corner case).

    Khác E-GraphSAGE/GCN/SAGE ở chỗ: 2 layer đầu dùng concat=True
    (multi-head) ⇒ số chiều = hidden_dim * heads. Layer cuối mean
    qua các head về hidden_dim để khớp input của head classifier.

    Phân loại vẫn chỉ trên E cạnh GỐC (không phải 2E cạnh MP) qua head
    GIỐNG các model khác: concat[h_u, h_v, edge_attr(gốc)] → MLP.

    Args
    ----
    edge_dim, num_classes, node_in_dim, hidden_dim, num_layers, dropout:
        cùng chữ ký với các model khác.
    heads : int, mặc định 4
        Số head attention cho MỖI layer (trừ layer cuối nếu có thể, layer
        cuối cũng dùng heads head nhưng concat=False để mean).

    Lưu ý
    -----
    - GAT có nhiều tham số hơn các baseline khác vì:
        (a) heads lần các ma trận trọng số cho node features.
        (b) thêm lin_edge = Linear(edge_dim, heads * out_channels).
        (c) vector attention att_src, att_dst, att_edge.
    - PyG’s GATv2Conv có sẵn dropout attention (mặc định 0.0) —
      để 0.0 cho reproducibility. Dropout model-level đặt giữa các
      layer qua self.dropout_between.
    - Cùng head classifier với 4 model kia để so sánh công bằng.
    """

    def __init__(self, edge_dim: int, num_classes: int,
                 node_in_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5,
                 heads: int = 4):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"GATBaseline: num_layers={num_layers} < 1.")
        if hidden_dim < 1:
            raise ValueError(f"GATBaseline: hidden_dim={hidden_dim} < 1.")
        if edge_dim < 1:
            raise ValueError(f"GATBaseline: edge_dim={edge_dim} < 1.")
        if num_classes < 1:
            raise ValueError(f"GATBaseline: num_classes={num_classes} < 1.")
        if heads < 1:
            raise ValueError(f"GATBaseline: heads={heads} < 1.")

        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.heads = heads

        self.dropout_between = nn.Dropout(p=dropout)

        # ----- GATv2Conv stack -----
        # PyG chú ý: truyền edge_attr=data.edge_attr_mp cho message
        # passing (2E cạnh MP).
        self.layers = nn.ModuleList()
        if num_layers == 1:
            # 1 layer duy nhất — phải concat=False để output ra hidden_dim.
            self.layers.append(
                GATv2Conv(node_in_dim, hidden_dim, heads=heads,
                          concat=False, edge_dim=edge_dim,
                          dropout=0.0)
            )
        else:
            # Layer 0: node feature → multi-head concat (hidden_dim * heads).
            self.layers.append(
                GATv2Conv(node_in_dim, hidden_dim, heads=heads,
                          concat=True, edge_dim=edge_dim,
                          dropout=0.0)
            )
            # Layers giữa (nếu có): vẫn concat=True.
            for _ in range(num_layers - 2):
                self.layers.append(
                    GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads,
                              concat=True, edge_dim=edge_dim,
                              dropout=0.0)
                )
            # Layer cuối: concat=False để mean các head về hidden_dim.
            self.layers.append(
                GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads,
                          concat=False, edge_dim=edge_dim,
                          dropout=0.0)
            )

        # ----- Edge classification head (GIỐNG 4 model khác) -----
        head_in_dim = hidden_dim * 2 + edge_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Khai báo trọng số Linear khởi tạo theo He.

        CHÚ Ý: PyG's GATv2Conv có lin_src / lin_dst / lin_edge
        cũng là nn.Linear, nên chúng bị override theo cùng quy ước với
        4 baseline khác. Các Parameter thuần (không phải Linear) như
        att_src, att_dst, att_edge KHÔNG bị override ở đây.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        """Forward cho edge classification. Trả về [E, num_classes].

        Message passing dùng edge_index_mp ([2, 2E]) + edge_attr_mp
        ([2E, F]); classification chỉ trên E cạnh GỐC.
        """
        x = data.x                                       # [N, node_in_dim]
        edge_index_mp = data.edge_index_mp               # [2, 2E]
        edge_attr_mp = data.edge_attr_mp                 # [2E, F]

        # === MESSAGE PASSING (2E cạnh, cả 2 chiều) ===
        h = x
        for layer in self.layers:
            h = layer(h, edge_index_mp, edge_attr_mp)    # GATv2Conv
            h = self.dropout_between(h)

        # === EDGE CLASSIFICATION (E cạnh gốc, cùng head 4 model kia) ===
        edge_index = data.edge_index                     # [2, E]
        edge_attr = data.edge_attr                       # [E, F]
        src = edge_index[0]
        dst = edge_index[1]
        h_u = h[src]
        h_v = h[dst]
        edge_repr = torch.cat([h_u, h_v, edge_attr], dim=-1)  # [E, 2H+F]
        logits = self.head(edge_repr)                           # [E, num_classes]
        return logits




def build_model(model_type: str, data, cfg: dict) -> torch.nn.Module:
    """
    Factory: khởi tạo model theo tên, đọc **động** chiều từ ``data``.

    KHÔNG hardcode bất kỳ số chiều nào — lấy từ:

        • ``data.feature_dim``    → ``edge_dim``
        • ``data.num_classes``    → ``num_classes``
        • ``data.x.shape[1]``     → ``node_in_dim``

    Hyperparameter lấy từ ``cfg['model']`` (đọc từ ``config.yaml``):

        • ``hidden_dim``
        • ``num_layers``
        • ``dropout``

    Parameters
    ----------
    model_type : str
        - ``'egraphsage'``         — E-GraphSAGE (MessagePassing tự viết).
        - ``'gcn'``                — baseline GCNConv (KHÔNG dùng edge feature).
        - ``'graphsage'``          — baseline SAGEConv (KHÔNG dùng edge feature).
        - ``'sage_edge_concat'``   — "Phương án B": nhồi edge vào node input.
    data : torch_geometric.data.Data
        Đồ thị đã qua ``build_graph``. Chỉ đọc metadata; không thay đổi.
    cfg : dict
        Cấu hình parse từ ``config.yaml`` (yaml.safe_load). Cần key
        ``model`` với ``hidden_dim``/``num_layers``/``dropout``.

    Returns
    -------
    torch.nn.Module
        Model ở trên CPU (chưa ``.to(device)``); ``train.py`` sẽ lo phần đó.
    """
    if not isinstance(cfg, dict) or 'model' not in cfg:
        raise ValueError(
            f"build_model: cfg phải là dict có key 'model'; got {type(cfg)}"
        )

    mcfg = cfg['model']
    edge_dim = int(data.feature_dim)
    num_classes = int(data.num_classes)
    node_in_dim = int(data.x.shape[1])

    common_kwargs = dict(
        edge_dim=edge_dim,
        num_classes=num_classes,
        node_in_dim=node_in_dim,
        hidden_dim=int(mcfg['hidden_dim']),
        num_layers=int(mcfg['num_layers']),
        dropout=float(mcfg['dropout']),
    )

    if model_type == 'egraphsage':
        return EGraphSAGE(**common_kwargs)
    if model_type == 'gcn':
        return GCNBaseline(**common_kwargs)
    if model_type == 'graphsage':
        return GraphSAGEBaseline(**common_kwargs)
    if model_type == 'sage_edge_concat':
        return SAGEEdgeConcatBaseline(**common_kwargs)
    if model_type == 'gat':
        return GATBaseline(**common_kwargs)
    raise ValueError(
        f"build_model: model_type='{model_type}' không hỗ trợ. "
        f"Chọn một trong: 'egraphsage', 'gcn', 'graphsage', "
        f"'sage_edge_concat', 'gat'."
    )
