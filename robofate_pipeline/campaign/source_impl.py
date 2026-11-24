"""Original task-budget and source assembly functions, externally bound."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from ..evidence import need

def task_budget(task):
    need(task in SPECS, 'Task binding has not been prepared')
    total, digest = SPECS[task]
    path = NATIVE / 'task/RoboDojo/tasks' / (task + '.py')
    raw = read_ref({'path': str(path), 'sha256': digest}, expected_path=path)
    values = [node.value.value for node in ast.walk(ast.parse(raw))
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
              and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                      and t.value.id == 'self' and t.attr == 'step_lim' for t in node.targets)]
    need(values == [total] and type(values[0]) is int, 'Actual task budget changed')
    return total, {'path': str(path), 'sha256': digest}


def assemble(task, layout_id, split_ref, split, total, task_ref, raw):
    need(task in SPECS and type(layout_id) is int, 'Explicit supported task and integer layout')
    need(total == SPECS[task][0] and type(total) is int, 'Native task budget mismatch')
    need(split['schema'] == 'refate.split_manifest.v1' and split['task'] == task
         and split['owner_chat'] == OWNER and split['campaign_id'] == sequence.CAMPAIGN, 'Source split identity mismatch')
    seq = split['initial_source_sequence']
    need(type(seq) is list and len(seq) == len(set(seq)) and all(type(i) is int for i in seq)
         and layout_id in seq and layout_id not in split['original_partition']['reserved_final'], 'Not in original collection sequence')
    row = split['layouts'][str(layout_id)]
    need(row['split'] in ('student_train', 'student_dev') and row['native_bank'] == 0
         and row['layout_id'] == layout_id, 'Wrong source/split')
    content = hashlib.sha256(original.canonical(raw).encode()).hexdigest()
    need(content == row['initialization_content_sha256'], 'Full initialization mismatch')
    need(row['initialization_distinct_from_preceding'] is True, 'Duplicate initialization cannot issue a new source')
    path = NATIVE / f'Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_{layout_id}.json'
    need(row['layout_ref']['path'] == str(path), 'Wrong raw source filename')
    need(_membership(split, layout_id, split_ref, raw), 'Group membership mismatch')
    source = dict(task=task, source_id=f'{task}_b0_l{layout_id}_{row["layout_ref"]["sha256"][:16]}',
                  source_owner=OWNER, execution_owner=OWNER, native_bank=0, layout_id=layout_id,
                  encoded_seed=layout_id, native_ordinal=layout_id,
                  resolved_layout_path=str(path.resolve()), layout_ref=deepcopy(row['layout_ref']),
                  layout_sha256=row['layout_ref']['sha256'], initialization_content_sha256=content,
                  group_id=row['group_id'], split=row['split'],
                  data_split={'student_train': 'train', 'student_dev': 'development'}[row['split']],
                  source_sequence=seq, required_accounted_prefix=seq[:seq.index(layout_id)])
    return dict(schema='refate.common_collector.source_preparation.v1', data_kind='preparation_only',
                binding_schema='refate.common_collector.task_source_binding.v1', source=source,
                mode='independent_baseline', native_step_limit=total, requested_initial_native_count=0,
                renderer_profile=PROFILE, prospective_teacher_renderer_profile=PROFILE,
                provenance={'split': split_ref, 'task_implementation': task_ref},
                source_admitted=False, source_reserved=False, launch_authority=False, qualification=False)
