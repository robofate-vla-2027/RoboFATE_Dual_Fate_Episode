"""Original camera counter checks with explicit task budgets."""
from dataclasses import dataclass

def need(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class CameraBudget:
    task_name: str
    total: int
    t1: int
    remaining: int
    task_totals: tuple

    def __post_init__(self):
        TASK_TOTALS = dict(self.task_totals)
        need(len(TASK_TOTALS) == len(self.task_totals) and TASK_TOTALS
             and all(type(k) is str and k and type(v) is int and v > 0
                     for k, v in self.task_totals), 'Invalid explicit task budgets')
        need(type(self.task_name) is str and self.task_name in TASK_TOTALS,
             'Unsupported camera task binding')
        need(all(type(n) is int for n in (self.total, self.t1, self.remaining))
             and (self.total, self.t1, self.remaining) ==
             (TASK_TOTALS[self.task_name], self.t1, TASK_TOTALS[self.task_name] - self.t1),
             'Camera task/native budget binding differs')
        need(0 < self.t1 < self.total, 'Nonterminal camera cutoff required')

    def index(self, native_index):
        need(type(native_index) is int and self.t1 <= native_index <= self.total,
             'Unsupported current native index for bound task')
        return self.total - native_index

    def actual_task(self, task, native_index):
        need(task.task_name == self.task_name and task.official_action_limit is True,
             'Camera task identity/official limit differs')
        need(all(type(n) is int and n == self.total for n in
                 (task.native.step_lim, task.native_step_limit, task.max_steps)),
             'Camera actual native budget differs')
        self.index(native_index)
