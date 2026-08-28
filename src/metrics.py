"""Đánh giá ViACaPu: nhãn lớp, tính precision/recall/F1, in báo cáo, và vòng eval.

Tách từ decode.py (repo gốc) — chỉ giữ phần metrics + eval dùng cho ViACaPu, không
kèm CLI/ONNX cũ. `evaluate()` chạy model trên một DataLoader và trả về F1 tổng thể
cùng (case_pred, case_true, punct_pred, punct_true) để dựng confusion matrix.
"""
import numpy as np
import torch

# !!! giữ khớp với tools/prepare_punct_case_data.py
punct_id = {0: "NO_PUNCT", 1: "COMMA", 2: "PERIOD", 3: "QUESTION"}
case_id = {0: "LOWER", 1: "UPPER", 2: "CAP", 3: "MIX_CASE"}


def inc(d, k):
    d[k] = d.get(k, 0) + 1


def get_metrics(output, target):
    """Trả về (precision, recall, f_scores, overall) theo lớp; overall micro trên lớp>0."""
    assert len(output) == len(target), f"output len:{output} != target len:{target}"
    true_predicted, all_predicted, all_expected = {}, {}, {}
    for i in range(len(output)):
        inc(all_expected, target[i])
        inc(all_predicted, output[i])
        if target[i] == output[i]:
            inc(true_predicted, output[i])

    precision = {k: (true_predicted.get(k, 0)) / all_predicted[k] for k in all_predicted}
    recall = {k: (true_predicted.get(k, 0)) / all_expected[k] for k in all_expected}
    f_scores = {
        k: None if precision[k] == 0 else (0 if recall[k] == 0
            else (2 * precision[k] * recall[k] / (precision[k] + recall[k])))
        for k in precision
    }

    otp = oap = oae = 0
    for k in all_expected:
        if k > 0:  # bỏ lớp 0 (NONE/LOWER) khỏi điểm tổng
            otp += true_predicted.get(k, 0)
            oap += all_predicted.get(k, 0)
            oae += all_expected[k]
    op = otp / oap if oap > 0 else 0
    orec = otp / oae if oae > 0 else 0
    of = 2 * op * orec / (op + orec) if orec > 0 else 0
    return precision, recall, f_scores, (op, orec, of)


def print_metrics(logging, precision, recall, f_scores, overall, label_map):
    for k in label_map:
        if k not in precision:
            continue
        logging.info(
            f"{label_map[k]}: \tPrec [{precision[k]:.3f}], "
            + (f"\tRec [{recall[k]:.3f}], " if k in recall else "\tRec [None], ")
            + (f"\tF1 [{f_scores[k]:.3f}], " if f_scores[k] is not None else "\tF1 [None], ")
        )
    logging.info(
        f"Overall: \tPrec [{overall[0]:.3f}], \tRec [{overall[1]:.3f}], \tF1 [{overall[2]:.3f}], "
    )


@torch.no_grad()
def evaluate(model, dl, dev):
    """Chạy model trên DataLoader, trả về (CaseF1, PunctF1, (cp, ct, pp, pt))."""
    from .data import to_dev
    model.eval()
    cp, ct, pp, pt = [], [], [], []
    for batch in dl:
        batch = to_dev(batch, dev)
        cl, pl, _ = model(batch)
        m = batch["score_mask"]
        cp.append(cl.argmax(-1)[m].cpu().numpy()); ct.append(batch["case"][m].cpu().numpy())
        pp.append(pl.argmax(-1)[m].cpu().numpy()); pt.append(batch["punct"][m].cpu().numpy())
    cp, ct = np.concatenate(cp), np.concatenate(ct)
    pp, pt = np.concatenate(pp), np.concatenate(pt)

    def overall_f1(pred, true):
        return get_metrics(pred, true)[3][2]
    return overall_f1(cp, ct), overall_f1(pp, pt), (cp, ct, pp, pt)
