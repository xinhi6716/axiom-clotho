"""Frozen one-attempt synthetic comparison. Prepare never draws study data or fits."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

STUDY_ID = 'CLOTHO-OPTIONAL-GROUPED-SYNTH-V1'
ROOTS = [731903,731927,731939,731951,731967]
SCENARIOS = ['linear_iid','smooth_iid','linear_domain_holdout']
FAMILIES = ['dummy','ridge','extra_trees','nystroem_ridge','group_median']
CONFIG = dict(study_id=STUDY_ID, roots=ROOTS, scenarios=SCENARIOS, families=FAMILIES,
              sizes=[2,3,4]*6, features=['x','z'], target='y', group='group',
              outer_splits=3, inner_splits=2, objective='group_mae', categorical_encoding='legacy-v1',
              main_cpu_seconds=200, total_cpu_seconds=300, wall_seconds=900,
              max_rss_bytes=512*1024**2, retained_bytes=16*1024**2, artifact_io_bytes=128*1024**2)
PROTOCOL_SHA = '708227ab3f634ffaed0ab5179e9189e4ea4f6019ea7e8c8e8eb159adcf41556a'
TREE_SHA = '2653ee2f361a443a0f2a87fa42ae497066e3c736'
COMMIT_SHA = '5033a191e00667e7dbc53055a3bbbda5dc2ebc19'
MANAGED_IO_BYTES = 0


def charge_io(size):
    global MANAGED_IO_BYTES
    MANAGED_IO_BYTES += size
    if MANAGED_IO_BYTES > CONFIG['artifact_io_bytes']:
        raise RuntimeError('Managed artifact IO budget exceeded.')


def read_bytes(path):
    value=Path(path).read_bytes(); charge_io(len(value)); return value


def read_text(path):
    return read_bytes(path).decode()


def digest(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def write_json(path,value):
    path=Path(path); data=encoded(value)
    charge_io(len(data))
    with path.open('xb') as handle: handle.write(data)
    return len(data)


def atomic_json(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    data=encoded(value)
    charge_io(len(data))
    with tmp.open('wb') as handle: handle.write(data)
    os.replace(tmp,path)
    return len(data)


def runtime():
    return {'python':platform.python_version(),'platform':platform.platform(),
            **{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','threadpoolctl']}}


def load_source(source):
    source=Path(source).resolve()
    sys.path.insert(0,str(source/'src'))
    import groupbench
    if Path(groupbench.__file__).resolve().parent != source/'src/groupbench':
        raise RuntimeError('Unexpected imported source identity.')
    return source


def source_hashes(source):
    return {str(p.relative_to(source)):digest(p) for p in sorted((source/'src/groupbench').rglob('*.py'))}


def roster():
    import numpy as np
    return np.repeat([f'g{i:02d}' for i in range(18)],CONFIG['sizes'])


def mapping(root):
    import numpy as np
    from groupbench.splitting import GroupKFold
    groups=roster(); folds=np.full(54,-1,dtype=int); saved=[]
    for fold,(train,test) in enumerate(GroupKFold(3,random_state=root).split(groups=groups)):
        folds[test]=fold
        saved.append({'fold':fold,'train':train.tolist(),'test':test.tolist(),
                      'test_groups':sorted(set(groups[test]))})
    if np.any(folds<0): raise RuntimeError('Incomplete prospective fold mapping.')
    return folds,saved


def prepare(source,protocol,out):
    """Zero-outcome placeholder plans; splitter RNG only, no study generator."""
    import numpy as np
    import pandas as pd
    from unittest.mock import patch
    source=load_source(source)
    from groupbench import plan_benchmark
    from groupbench.evaluation import _prepare_benchmark
    from groupbench.model import GroupBenchRegressor
    if digest(protocol)!=PROTOCOL_SHA: raise ValueError('Protocol pin differs.')
    source_receipt=source.parent/'manifest.json'
    receipt=json.loads(read_text(source_receipt))
    if receipt.get('final_union_tree_sha')!=TREE_SHA:raise RuntimeError('Root-reviewed source tree receipt differs.')
    for name,sha in source_hashes(source).items():
        if receipt['files'].get(name,{}).get('sha256')!=sha:raise RuntimeError('Source file differs from reviewed union.')
    out=Path(out); out.mkdir(parents=False,exist_ok=False)
    frame=pd.DataFrame({'x':np.zeros(54),'z':np.zeros(54),'y':np.zeros(54),'group':roster()})
    cases=[]
    with patch.object(GroupBenchRegressor,'_fit_prepared',side_effect=AssertionError('BUILD must not fit')):
        for scenario_id,scenario in enumerate(SCENARIOS):
            for root in ROOTS:
                fold_ids,folds=mapping(root)
                plan=plan_benchmark(frame,target='y',group='group',features=['x','z'],
                                    baselines=tuple(FAMILIES),outer_splits=3,inner_splits=2,
                                    random_state=root,objective='group_mae')
                if plan['fit_budget']['total_fits']!=75: raise RuntimeError('Unexpected fit schedule.')
                if plan['fit_budget']['inner_selection_fits']!=60 or plan['fit_budget']['outer_family_refits']!=15:
                    raise RuntimeError('Unexpected fit breakdown.')
                _,_,_,_,schedule=_prepare_benchmark(frame,target='y',group='group',features=['x','z'],
                    baselines=tuple(FAMILIES),outer_splits=3,inner_splits=2,random_state=root,objective='group_mae')
                for fold,(train,test,selection) in enumerate(schedule):
                    if train.tolist()!=folds[fold]['train'] or test.tolist()!=folds[fold]['test']:
                        raise RuntimeError('Prospective domain mapping differs from benchmark schedule.')
                for fold,entry in enumerate(plan['outer_folds']):
                    if entry['test_rows']!=len(folds[fold]['test']): raise RuntimeError('Outer mapping mismatch.')
                cases.append({'case_id':f'{scenario_id}_{root}','scenario_id':scenario_id,'scenario':scenario,
                              'root':root,'fold_ids':fold_ids.tolist(),'folds':folds,'plan':plan,
                              'fold_roles':['left_extrapolation','interpolation','right_extrapolation']
                                           if scenario_id==2 else ['iid_group_holdout']*3})
    if sum(c['plan']['fit_budget']['total_fits'] for c in cases)!=1125: raise RuntimeError('Total differs.')
    write_json(out/'config.json',CONFIG);write_json(out/'cases.json',cases)
    write_json(out/'runtime.json',runtime());write_json(out/'source_hashes.json',source_hashes(source))
    manifest={'study_id':STUDY_ID,'status':'PREPARED_NO_STUDY_DRAWS_OR_FITS','source':str(source),
              'source_commit_sha':COMMIT_SHA,'source_tree_sha':TREE_SHA,'source_review_manifest_sha256':digest(source_receipt),'protocol_sha256':PROTOCOL_SHA,'driver_sha256':digest(__file__),
              'files':{name:digest(out/name) for name in ['config.json','cases.json','runtime.json','source_hashes.json']},
              'expected_cases':15,'expected_fits':1125,'study_data_draws':0,'study_fits':0,
              'splitter':'groupbench.splitting.GroupKFold(3, random_state=root); no shuffle argument exists'}
    write_json(out/'PREPARED.json',manifest)
    return {'status':manifest['status'],'prepared_sha256':digest(out/'PREPARED.json'),'expected_fits':1125}


def generate_case(scenario_id,root,fold_ids,*,rng=None):
    """Only execution invokes real PCG64. Tests inject a deterministic fake RNG."""
    import numpy as np
    import pandas as pd
    if scenario_id not in range(3) or root not in ROOTS or len(fold_ids)!=54:
        raise ValueError('Case is outside the frozen protocol.')
    if any(type(v) not in (int,np.int64,np.int32) or int(v) not in (0,1,2) for v in fold_ids):
        raise ValueError('Invalid prospective fold IDs.')
    if rng is None: rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([root,scenario_id])))
    values=[];position=0
    for group_id,size in enumerate(CONFIG['sizes']):
        b=rng.normal(0,0.2)
        fold=int(fold_ids[position])
        bounds=[(-2,-1),(-.5,.5),(1,2)][fold] if scenario_id==2 else (-1,1)
        x=rng.uniform(*bounds,size=size)
        z=rng.uniform(-1,1,size=size)
        epsilon=rng.normal(0,.15,size=size)
        f=1.5+2*np.sin(np.pi*x)+.5*z*z if scenario_id==1 else 1.5+2*x+.5*z
        y=f+b+epsilon
        for i in range(size):values.append((float(x[i]),float(z[i]),float(y[i]),f'g{group_id:02d}'))
        position+=size
    frame=pd.DataFrame(values,columns=['x','z','y','group'])
    if len(frame)!=54 or not np.isfinite(frame[['x','z','y']].to_numpy()).all():
        raise RuntimeError('Invalid generated cohort.')
    return frame


def validate_pins(prepared):
    prepared=Path(prepared);m=json.loads(read_text(prepared/'PREPARED.json'))
    if m['driver_sha256']!=digest(__file__) or m['protocol_sha256']!=PROTOCOL_SHA:
        raise RuntimeError('Frozen driver/protocol changed.')
    for name,sha in m['files'].items():
        if digest(prepared/name)!=sha:raise RuntimeError('Prepared bytes changed.')
    if json.loads(read_text(prepared/'config.json'))!=CONFIG:raise RuntimeError('Configuration differs.')
    source=load_source(m['source'])
    if source_hashes(source)!=json.loads(read_text(prepared/'source_hashes.json')):
        raise RuntimeError('Integrated source changed.')
    if runtime()!=json.loads(read_text(prepared/'runtime.json')):raise RuntimeError('Runtime changed.')
    return m


def execute_child(prepared):
    """Runs once only after supervisor validates release and creates attempt marker."""
    prepared=Path(prepared);m=validate_pins(prepared)
    if not (prepared/'ATTEMPT.json').is_file():raise RuntimeError('Supervisor attempt marker absent.')
    write_json(prepared/'CHILD_STARTED.json',{'status':'ONE_CHILD_STARTED','started_unix':time.time()})
    import numpy as np
    from sklearn.pipeline import Pipeline
    from unittest.mock import patch
    from groupbench import benchmark
    cases=json.loads(read_text(prepared/'cases.json'))
    stats={'attempted_fits':0,'completed_fits':0,'failed_fits':0,'datasets_started':0,'datasets_drawn':0,'cases_sealed':0,'managed_artifact_bytes_written':0}
    root=prepared/'outputs';root.mkdir(exist_ok=False)
    original=Pipeline.fit
    def progress():
        stats['managed_artifact_io_bytes']=MANAGED_IO_BYTES
        stats['managed_artifact_bytes_written']+=atomic_json(prepared/'PROGRESS.json',stats)
        if stats['managed_artifact_bytes_written']>CONFIG['artifact_io_bytes']:raise RuntimeError('Artifact IO budget exceeded.')
    def tracked(pipeline,X,y=None,**kwargs):
        top='model' in pipeline.named_steps
        if top:stats['attempted_fits']+=1;progress()
        try:value=original(pipeline,X,y,**kwargs)
        except BaseException:
            if top:stats['failed_fits']+=1;progress()
            raise
        if top:stats['completed_fits']+=1;progress()
        return value
    sealed=[];started=time.monotonic()
    try:
        with patch.object(Pipeline,'fit',tracked):
            for case in cases:
                before=stats['completed_fits'];out=root/case['case_id'];out.mkdir(exist_ok=False)
                stats['datasets_started']+=1;progress()
                frame=generate_case(case['scenario_id'],case['root'],case['fold_ids'])
                stats['datasets_drawn']+=1;progress()
                # Exact numerical inputs remain available without object-pickle loading.
                np.savez(out/'data.npz',x=frame.x.to_numpy(),z=frame.z.to_numpy(),y=frame.y.to_numpy(),group=frame.group.to_numpy(dtype='U3'))
                charge_io((out/'data.npz').stat().st_size)
                stats['managed_artifact_bytes_written']+=(out/'data.npz').stat().st_size
                result=benchmark(frame,target='y',group='group',features=['x','z'],baselines=tuple(FAMILIES),
                                 outer_splits=3,inner_splits=2,random_state=case['root'],objective='group_mae')
                if stats['completed_fits']-before!=75:raise RuntimeError('Actual per-case fit count differs.')
                if result.predictions.outer_fold.tolist()!=case['fold_ids']:raise RuntimeError('Actual outer mapping differs.')
                for name,value in [('manifest.json',result.manifest),('summary.json',result.summary)]:
                    stats['managed_artifact_bytes_written']+=write_json(out/name,value)
                raw=result.predictions.to_csv(index=False).encode()
                charge_io(len(raw))
                with (out/'predictions.csv').open('xb') as f:f.write(raw)
                stats['managed_artifact_bytes_written']+=len(raw)
                files={f.name:digest(f) for f in sorted(out.iterdir()) if f.is_file()}
                sealed.append({'case_id':case['case_id'],'files':files,'fits_completed':75})
                stats['managed_artifact_bytes_written']+=write_json(out/'SEALED.json',sealed[-1])
                stats['cases_sealed']+=1;progress()
                if sum(f.stat().st_size for f in prepared.rglob('*') if f.is_file())>CONFIG['retained_bytes']:
                    raise RuntimeError('Retained artifact budget exceeded.')
        if stats['attempted_fits']!=1125 or stats['completed_fits']!=1125 or stats['failed_fits']:
            raise RuntimeError('Actual total fit counts differ.')
        write_json(prepared/'ALL_CASES_SEALED.json',{'status':'ALL_15_SEALED_BEFORE_OPERATOR_COMPARISON',
                   'cases':sealed,'stats':stats,'wall_seconds':time.monotonic()-started,
                   'disclosure':'benchmark computed metrics internally; operator did not inspect or render intermediate results'})
    except BaseException as exc:
        atomic_json(prepared/'FAILURE.json',{'status':'INCOMPLETE_STOPPED_NO_RETRY','exception_type':type(exc).__name__,
                    'message':str(exc),'stats':stats})
        raise


def supervised_run(prepared,release):
    prepared=Path(prepared);validate_pins(prepared)
    grant=json.loads(read_text(release))
    expected={'study_id':STUDY_ID,'action':'RUN_ONCE','prepared_sha256':digest(prepared/'PREPARED.json'),
              'driver_sha256':digest(__file__),'protocol_sha256':PROTOCOL_SHA}
    if grant!=expected:raise ValueError('Exact root-reviewed RUN release required.')
    write_json(prepared/'ATTEMPT.json',{'status':'ONE_ATTEMPT_STARTED','release_sha256':digest(release),
                                      'started_unix':time.time(),'prepared_sha256':expected['prepared_sha256']})
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def limits():
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_CPU,(CONFIG['main_cpu_seconds'],CONFIG['main_cpu_seconds']+1))
    start=time.monotonic();start_cpu=time.process_time();reason=None;peak_rss=0;cpu=0;child_io=0
    with (prepared/'EXECUTION.stdout').open('xb') as stdout,(prepared/'EXECUTION.stderr').open('xb') as stderr:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_execute',str(prepared)],
                              env=env,stdout=stdout,stderr=stderr,start_new_session=True,preexec_fn=limits)
        while proc.poll() is None:
            # The implementation does not spawn estimator worker processes; flag any
            # unexpected descendant rather than under-accounting an unknown topology.
            try:
                children=Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
                if children:reason='unexpected_child_process_topology'
                fields=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()
                cpu=(int(fields[11])+int(fields[12])+int(fields[13])+int(fields[14]))/os.sysconf('SC_CLK_TCK')
                status=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                rss=next((int(x.split()[1])*1024 for x in status if x.startswith('VmRSS:')),0)
                peak_rss=max(peak_rss,rss)
                if (prepared/'PROGRESS.json').exists():
                    child_io=json.loads(read_text(prepared/'PROGRESS.json')).get('managed_artifact_io_bytes',0)
                if child_io+MANAGED_IO_BYTES>CONFIG['artifact_io_bytes']:reason='managed_io_budget'
                if cpu>CONFIG['main_cpu_seconds']:reason='main_cpu_budget'
                if rss>CONFIG['max_rss_bytes']:reason='rss_budget'
                if time.monotonic()-start>CONFIG['wall_seconds']:reason='wall_budget'
                if sum(f.stat().st_size for f in prepared.rglob('*') if f.is_file())>CONFIG['retained_bytes']:reason='retained_budget'
            except (FileNotFoundError,ProcessLookupError):pass
            if reason:
                os.killpg(proc.pid,signal.SIGKILL);break
            time.sleep(.05)
        returncode=proc.wait()
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    if usage.ru_utime+usage.ru_stime>CONFIG['main_cpu_seconds']:reason=reason or 'main_cpu_budget_postrun'
    if usage.ru_maxrss*1024>CONFIG['max_rss_bytes']:reason=reason or 'rss_budget_postrun'
    if (prepared/'ALL_CASES_SEALED.json').is_file():
        child_io=json.loads(read_text(prepared/'ALL_CASES_SEALED.json'))['stats'].get('managed_artifact_io_bytes',child_io)
    if child_io+MANAGED_IO_BYTES>CONFIG['artifact_io_bytes']:reason=reason or 'managed_io_budget_postrun'
    success=returncode==0 and reason is None and (prepared/'ALL_CASES_SEALED.json').is_file()
    write_json(prepared/'SUPERVISOR.json',{'returncode':returncode,'stop_reason':reason,'wall_seconds':time.monotonic()-start,
               'sampled_peak_rss_bytes':peak_rss,'child_rusage_max_rss_kib':usage.ru_maxrss,
               'child_cpu_seconds':usage.ru_utime+usage.ru_stime,'supervisor_cpu_seconds':time.process_time()-start_cpu,
               'managed_artifact_io_bytes_observed':child_io+MANAGED_IO_BYTES,
               'io_scope':'explicit driver artifact/source reads and writes; excludes Python/dependency loader OS traffic','topology':'single worker process; unexpected descendants stop the attempt',
               'status':'SEALED' if success else 'INCOMPLETE_NO_RETRY'})
    return {'returncode':returncode,'status':'SEALED' if success else 'INCOMPLETE_NO_RETRY'}


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',required=True);p.add_argument('--protocol',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('run');p.add_argument('--prepared',required=True);p.add_argument('--release',required=True)
    p=sub.add_parser('_execute');p.add_argument('prepared')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.source,args.protocol,args.out)
    elif args.command=='run':result=supervised_run(args.prepared,args.release)
    else:execute_child(args.prepared);result={'status':'ALL_CASES_SEALED'}
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
