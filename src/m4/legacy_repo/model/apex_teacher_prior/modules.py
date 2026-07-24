import torch as pt
import torch.nn.functional as F


class TeacherHiddenAdapter(pt.nn.Module):
    """Gated teacher hidden fusion.

    The student remains the main representation. Teacher hidden states only add
    a bounded residual prior through an adapter and a learned gate.
    """

    def __init__(self, student_dim, teacher_dim, dropout=0.1):
        super().__init__()
        self.teacher_dim = int(teacher_dim)
        self.adapter = pt.nn.Sequential(
            pt.nn.LayerNorm(self.teacher_dim),
            pt.nn.Linear(self.teacher_dim, student_dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(student_dim, student_dim),
        )
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(student_dim + self.teacher_dim),
            pt.nn.Linear(student_dim + self.teacher_dim, student_dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(student_dim, student_dim),
            pt.nn.Sigmoid(),
        )
        self.norm = pt.nn.LayerNorm(student_dim)

    def forward(self, h_student, teacher_hidden):
        if teacher_hidden is None:
            return h_student
        teacher_hidden = teacher_hidden.to(device=h_student.device, dtype=h_student.dtype)
        if teacher_hidden.shape[-1] < self.teacher_dim:
            pad = teacher_hidden.new_zeros((*teacher_hidden.shape[:-1], self.teacher_dim - teacher_hidden.shape[-1]))
            teacher_hidden = pt.cat([teacher_hidden, pad], dim=-1)
        elif teacher_hidden.shape[-1] > self.teacher_dim:
            teacher_hidden = teacher_hidden[..., : self.teacher_dim]
        gate = self.gate(pt.cat([h_student, teacher_hidden], dim=-1))
        return self.norm(h_student + gate * self.adapter(teacher_hidden))


def teacher_logit_distillation_loss(student_logits, teacher_logits, mask, weight=0.0, mode="mse", temperature=1.0):
    weight = float(weight)
    if weight <= 0 or teacher_logits is None:
        return student_logits.new_tensor(0.0)
    valid = mask > 0.5
    if not pt.any(valid):
        return student_logits.new_tensor(0.0)
    s = student_logits[valid].float()
    t = teacher_logits.to(device=student_logits.device, dtype=student_logits.dtype)[valid].float()
    if mode == "bce":
        target = pt.sigmoid(t / max(float(temperature), 1e-6))
        loss = F.binary_cross_entropy_with_logits(s, target)
    else:
        loss = F.mse_loss(s / max(float(temperature), 1e-6), t / max(float(temperature), 1e-6))
    return weight * loss

