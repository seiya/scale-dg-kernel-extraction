#!/usr/bin/env python3
"""Rebuild the two uniform Fig. 4 CSVs from one campaign directory.

The FLOP column is deliberately parsed from reports/README.md's authoritative
``FLOP / RK stage`` table, rather than duplicated as a second hand-maintained
table.  nsys intervals use volume_flux launch starts; all summaries retain
unrounded samples in the output CSVs.

volume_flux_us_per_stage is the MEDIAN of the per-launch durations over the
kept launches (measurement_inventory.md section 28.4 uses the mean, i.e. the
sum over launches divided by the stage count; at p=127 the two differ by
about 1%).  The category sums (gemm_, rk_update_, ...) are means.

device_us_per_stage divides the "CUDA device ..." sum of a run by the number
of tendency calls that sum actually covers, which is NOT 3*steps for the
logs written before 2026-09-16.  The tendency wrappers read their CUDA event
pair back one call late (mod_cuda_dg_kernels.cuf, "Device-time
measurement"), so when the time loop reset the timers after the warm-up steps
the measurement of the last warm-up call was still pending; the first call of
the window added it to the sum, and setup_advect3d_eq_finalize flushed the
last call's measurement as well.  With nwarmup >= 1 the sum over N = 3*steps
measured stages therefore covers N+1 intervals (with nwarmup = 0 nothing is
pending at the reset and it covers N).  Executables built from the commit
that fixed this print "Device-time calls summed:" next to the "CUDA device
..." lines; that count is used when present, and the N+1 / N rule when it
is not.  The A/B ratios of one campaign (fold on/off, GEMM_FUSED/FUSED_TC)
share N and are unaffected; only the absolute per-stage times move.
"""
import argparse, csv, re, statistics, pathlib

NQ = {7:8,15:16,31:32,63:64,127:128,255:256,511:512,575:576,767:768,1023:1024}

def med(a): return statistics.median(a)
def rng(a): return (min(a), max(a))
def metadata(run):
    d = {}
    for line in (run/'meta/metadata.txt').read_text().splitlines():
        for k,v in re.findall(r'([A-Za-z0-9_]+)=([^\s]+)', line): d[k]=v
    return d
def flops_from_readme(root):
    text=(root/'reports/README.md').read_text()
    # Restrict to the common-work table, terminating before B/node side table.
    text=text[text.index('| p | Nq | Ne | 体積点 | FLOP / RK stage'):]
    out={}
    for line in text.splitlines():
        m=re.match(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|.*?\|\s*([0-9.]+e[0-9]+)\s*\|', line)
        if m: out[int(m.group(1))]=float(m.group(3))
        if len(out)==10: break
    if set(NQ)-set(out): raise RuntimeError('README FLOP table parse incomplete')
    return out
def device_calls_summed(run, d):
    """Number of tendency calls the run's "CUDA device ..." sum covers."""
    logs=list(run.glob(f"*/r{d['round']:02d}_p{d['p']}_{d['path']}_{d['mode']}.log"))
    if len(logs) != 1:
        raise RuntimeError(f'expected one log for {d}, found {logs}')
    text=logs[0].read_text(errors='replace')
    m=re.search(r'Device-time calls summed:\s*(\d+)', text)
    if m: return int(m.group(1))
    m=re.search(r'Skipped warm-up steps:\s*(\d+)', text)
    if not m: raise RuntimeError(f'no warm-up count in {logs[0]}')
    nwarmup=int(m.group(1))
    # Pre-fix executable: the pending warm-up measurement was summed too.
    return 3*int(d['steps']) + (1 if nwarmup >= 1 else 0)
def raw(run):
    rows=[]
    for line in (run/'raw_results.txt').read_text().splitlines():
        d=dict(re.findall(r'(\w+)=([^\s]+)',line)); d['p']=int(d['p']); d['round']=int(d['round'])
        if int(d['rc']) != 0 or int(d['steps']) <= 0:
            raise RuntimeError(f"failed or empty run: {line}")
        d['device_calls_summed']=device_calls_summed(run, d)
        d['device_us_per_stage']=float(d['device_sum_s'])*1e6/d['device_calls_summed']
        rows.append(d)
    return rows
def expect_rounds(rows, *, section, p, path, mode=None):
    rs=[r for r in rows if r['section']==section and r['p']==p and r['path']==path
        and (mode is None or r['mode']==mode)]
    rounds=sorted(r['round'] for r in rs)
    if rounds != list(range(1,11)):
        raise RuntimeError(f'incomplete rounds for {section} p={p} path={path} mode={mode}: {rounds}')
    return rs
def trace(csvpath):
    # nsys writes two prose lines before CSV.  Parse by locating its header.
    lines=csvpath.read_text(errors='replace').splitlines()
    i=next(i for i,x in enumerate(lines) if x.startswith('Start (ns),'))
    rows=list(csv.DictReader(lines[i:]))
    x=[]
    for r in rows:
        try: start=float(r['Start (ns)']); dur=float(r['Duration (ns)']); name=r['Name']
        except (KeyError,ValueError): continue
        x.append((start,dur,name))
    return x
def classify(name):
    n=name.lower()
    if 'volume_flux_kernel' in n:return 'volume_flux'
    if 'elembnd_flux_kernel' in n:return 'side_elembnd'
    if 'rk_update' in n:return 'rk_update'
    if 'update_halo' in n:return 'update_halo'
    if 'gemm' in n or 'cutlass::kernel' in n:return 'gemm'
    return 'residual'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True,type=pathlib.Path); ap.add_argument('--fold-run-dir',required=True,type=pathlib.Path); ap.add_argument('--paper-data-dir',required=True,type=pathlib.Path); args=ap.parse_args()
    run=args.run_dir.resolve(); fold_run=args.fold_run_dir.resolve(); root=pathlib.Path(__file__).resolve().parents[1]
    meta=metadata(run); fold_meta=metadata(fold_run); flops=flops_from_readme(root); rows=raw(run); fold_rows=raw(fold_run)
    if fold_meta.get('fixed_ScaleDG_ZS3') != '0' or fold_meta.get('fixed_z_stages') != '4' or fold_meta.get('fixed_ScaleDG_XYOVL') != '0':
        raise RuntimeError('fold rerun is not the required fixed four-stage, no-overlap A/B')
    if fold_meta.get('binary_sha256') != meta.get('binary_sha256'):
        raise RuntimeError('timing/trace and fold rerun used different binaries')
    args.paper_data_dir.mkdir(parents=True,exist_ok=True)
    # Fig 4b: one row per timing round, plus per-cell median/range repeated for easy plotting.
    b=[]
    for p in (7,15,31,63,127,255,511):
      for path in ('CUDAFORTRAN_GEMM_FUSED','CUDAFORTRAN_FUSED_TC'):
        rs=expect_rounds(rows,section='timing',p=p,path=path)
        vals=[r['device_us_per_stage'] for r in rs]; m=med(vals); lo,hi=rng(vals)
        for r in rs:
          eff=flops[p]/(r['device_us_per_stage']*1e-6)/1e12
          b.append(dict(commit=meta.get('commit',''),binary_sha256=meta.get('binary_sha256',''),job=meta.get('job',''),node=meta.get('node',''),p=p,Nq=NQ[p],path=path,round=r['round'],measured_steps=int(r['steps']),device_calls_summed=r['device_calls_summed'],device_us_per_stage=r['device_us_per_stage'],median_us_per_stage=m,min_us_per_stage=lo,max_us_per_stage=hi,flops_per_stage=flops[p],effective_tflops=eff,percent_of_40p1=100*eff/40.1,lead_pt=''))
    for p in (7,15,31,63,127,255,511):
      gf=med([r['device_us_per_stage'] for r in expect_rounds(rows,section='timing',p=p,path='CUDAFORTRAN_GEMM_FUSED')])
      tc=med([r['device_us_per_stage'] for r in expect_rounds(rows,section='timing',p=p,path='CUDAFORTRAN_FUSED_TC')])
      lead=100*(flops[p]/(tc*1e-6)/1e12)/40.1-100*(flops[p]/(gf*1e-6)/1e12)/40.1
      for r in b:
        if r['p']==p:r['lead_pt']=lead
    # Fig 4a: nsys production trace and device-event fold-only A/B.
    a=[]
    for p in (127,255,511,575,767,1023):
      ev=trace(run/'nsys'/f'p{p}_gemm_fused.csv'); vol=[x for x in ev if classify(x[2])=='volume_flux']
      # Leave first three stages out as initialization; use the same suffix for durations.
      if len(vol) != 72:
        raise RuntimeError(f'unexpected volume-flux launch count for p={p}: {len(vol)}')
      vol=vol[3:]; periods=[(vol[i+1][0]-vol[i][0])/1e3 for i in range(len(vol)-1)]
      cats={}
      first,last=vol[0][0],vol[-1][0]
      # Kernels launched in [first, last) belong to the 68 complete stage
      # periods; the last volume_flux launch opens a stage whose GEMMs start
      # after `last`, so a closed window would count 69 flux kernels against
      # 68 stages of GEMMs.  Category sums are therefore per period.
      for st,du,name in ev:
        if first <= st < last: cats.setdefault(classify(name),[]).append(du/1e3)
      nper=len(periods)
      on_rows=expect_rounds(fold_rows,section='fold_fixed4',p=p,path='CUDAFORTRAN_GEMM_FUSED',mode='on')
      off_rows=expect_rounds(fold_rows,section='fold_fixed4',p=p,path='CUDAFORTRAN_GEMM_FUSED',mode='off')
      on=[r['device_us_per_stage'] for r in on_rows]; off=[r['device_us_per_stage'] for r in off_rows]
      # The on/off ratio is independent of the divisor only if both share it.
      divs={r['device_calls_summed'] for r in on_rows+off_rows}
      if len(divs) != 1: raise RuntimeError(f'fold on/off rounds at p={p} summed different call counts: {divs}')
      vm=med([x[1]/1e3 for x in vol]); pm=med(periods)
      # Main-stream kernel sums per period (side-stream elembnd and the
      # unclassified kernels, which may sit on other streams, excluded); the
      # period residual is what the main stream's kernels do not account for
      # (launch gaps and the fork/join tail).  The sums are means and the
      # period is a median, so the residual can come out slightly negative.
      main=sum(sum(cats.get(k,[])) for k in ('volume_flux','gemm','rk_update','update_halo'))/nper
      a.append(dict(commit=meta.get('commit',''),binary_sha256=meta.get('binary_sha256',''),trace_job=meta.get('job',''),trace_node=meta.get('node',''),fold_job=fold_meta.get('job',''),fold_node=fold_meta.get('node',''),p=p,Nq=NQ[p],fusion_stage_count=len(vol),fusion_period_count=len(periods),volume_flux_us_per_stage=vm,median_stage_period_us=pm,fusion_prize_percent=100*vm/pm,fold_fixed_z_stage_count=4,fold_device_calls_summed=divs.pop(),fold_on_round_us=';'.join(map(str,on)),fold_off_round_us=';'.join(map(str,off)),fold_on_median_us=med(on),fold_on_min_us=min(on),fold_on_max_us=max(on),fold_off_median_us=med(off),fold_off_min_us=min(off),fold_off_max_us=max(off),fold_gain_percent=100*(med(off)-med(on))/med(off),side_elembnd_us_per_stage=sum(cats.get('side_elembnd',[]))/nper,gemm_us_per_stage=sum(cats.get('gemm',[]))/nper,rk_update_us_per_stage=sum(cats.get('rk_update',[]))/nper,update_halo_us_per_stage=sum(cats.get('update_halo',[]))/nper,unclassified_kernel_us_per_stage=sum(cats.get('residual',[]))/nper,period_residual_us_per_stage=pm-main))
    for path,data in ((args.paper_data_dir/'fig4a_uniform.csv',a),(args.paper_data_dir/'fig4b_uniform.csv',b)):
      with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n'); w.writeheader(); w.writerows(data)
if __name__=='__main__': main()
