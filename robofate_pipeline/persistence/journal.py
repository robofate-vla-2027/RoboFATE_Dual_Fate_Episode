"""Original teacher proposal/committed-transition journal."""
import copy
from pathlib import Path
import time
from ..recording import teacher as common_teacher
from ..recording.common import need
from ..evidence import file_ref

class Journal:
    def __init__(self,api,*,run,identity,task,recorder,core,causal_history,source_run,t1,total,authorized_root,camera_budget):
        self.api,self.run,self.task,self.recorder=api,Path(run),task,recorder
        self.identity=copy.deepcopy(identity);self.t1=t1;self.total=total;self.sequence=0
        self.source=Path(source_run);self.pending=None
        need(task.control_steps==t1 and len(causal_history)==t1+1,'Journal must bind current actual t1')
        ipc=self.run/'teacher_ipc';ipc.mkdir()
        self.files=api.peer.MainFiles(ipc,authorized_root=authorized_root,camera_budget=camera_budget)
        self.host=api.host.Files(ipc,authorized_root=authorized_root)
        self.files.bind_recorder(recorder,core.wait_for_native_publication)
        history=[]
        for i,item in enumerate(causal_history):
            need(item['native_step']==i,'Actual history sequence changed')
            observation=copy.deepcopy(item['observation'])
            observation.pop('vision')
            folder=self.source/('initial' if i==0 else f'macro_obs_{i:06d}')
            history.append(dict(native_step=i,executed_action=api.plain(item['executed_action']),
                feedback=api.plain(item['feedback']),public_nonvisual=api.plain(observation),
                source_observation=file_ref(folder/'observation.json'),
                source_arrays=file_ref(folder/'arrays.npz'),
                source_rgb={c:file_ref(folder/(c+'.png')) for c in common_teacher.CAMERAS}))
        self.history_ref=self.files.publish('artifacts/causal_history.json',history)
        self.last=self.boundary(0)
        self.files.publish('bootstrap.json',dict(transport='in_process_original_Model',
            **self.identity,observation_ref=self.last,causal_history_ref=self.history_ref,
            native_used=t1,remaining_steps=total-t1,wall_ns=time.time_ns()))

    def boundary(self,sequence):
        used=self.task.control_steps
        need(used==self.t1+sequence,'Observation not at current executed boundary')
        descriptor=self.files.public_observation(sequence,self.task.public_observation())
        return self.files.publish(f'observation_{sequence:06d}.json',dict(**self.identity,
            native_used=used,native_limit=self.total,remaining_steps=self.total-used,
            observation=descriptor,transport='in_process_original_Model',wall_ns=time.time_ns()))

    def proposal(self,message):
        need(self.pending is None,'Previous proposal not paired to actual execution')
        seq=self.sequence+1;meta=message['refate']
        need(message['kind']=='action' and meta['source_id']==self.identity['source_id']
             and meta['sequence']==seq and meta['native_before']==self.task.control_steps==self.t1+seq-1,
             'Actual model proposal sequence/source differs')
        request=self.files.publish(f'request_{seq:06d}.json',dict(**self.identity,sequence=seq,
            observation_ref=self.last,transport='in_process_original_Model.get_takeover_request',wall_ns=time.time_ns()))
        response=dict(self.api.plain(message),sequence=seq,observation_ref=self.last,
                      request_sha256=request['sha256'],transport='in_process_original_Model_return',wall_ns=time.time_ns())
        self.host.publish(f'response_{seq:06d}.json',response)
        self.pending=copy.deepcopy(message)

    def committed(self,recorder,*,before,after,event):
        need(recorder is self.recorder and self.pending is not None,'Missing actual pending model action')
        need(event['action']==self.api.plain(self.pending['action'])
             and before['native_used']==self.t1+self.sequence and after['native_used']==before['native_used']+1,
             'Executed action/counter differs from saved proposal')
        self.sequence+=1;self.last=self.boundary(self.sequence)
        segment=dict(native_before=before['native_used'],native_after=after['native_used'],
            actual_control_range=[10*before['native_used']+1,10*after['native_used']],
            dense_path=str(self.run/'dense/control_steps.jsonl'),actual_event=event,
            observation_after_ref=self.last,wall_ns=time.time_ns())
        r=self.files.publish(f'artifacts/transition_{self.sequence:06d}.json',segment)
        self.pending=None
        return str(self.files.root/r['path'])+'#sha256='+r['sha256']

    def attach(self,model):
        original=model.get_takeover_request
        def observed():
            message=original();self.proposal(message);return message
        model.get_takeover_request=observed
