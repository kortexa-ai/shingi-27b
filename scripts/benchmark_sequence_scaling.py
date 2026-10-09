"""Measure identical isolated prompts on a private native worker with wider sequences.

The fixture contains complete native requests. No service is stopped by this script.
Use the matching experimental readout; production limits stay at four sequences.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import threading
import time

from shingi.decision import softmax
from shingi.gpu import gpu_free_mib, gpu_snapshot


def measure(executable, model, projector, fixture, slots, batch_tokens, rounds, out,
            reference=None, profile=False, limit=None, length_order=False):
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads(fixture.read_text())
    items = data['items'][:limit] if limit else data['items']
    expected = None
    if reference:
        expected = json.loads(reference.read_text())['runs'][0]['probabilities']
    report = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
              'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
              'slots': slots, 'batch_tokens': batch_tokens, 'runs': [], 'controls': [],
              'memory_samples': [], 'profiled': profile, 'items': len(items), 'length_order': length_order}
    def save():
        (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    # The wrapper also monitors the whole GPU; this is a local serving headroom check.
    assert gpu_free_mib() >= 32 * 1024, 'preload headroom'
    env = dict(os.environ, SHINGI_TRIAL_BATCH_TOKENS=str(batch_tokens))
    argv = [str(executable), str(model), '16384'] + ([str(projector)] if projector else [])
    argv += ['--parallel', str(slots), '--prefix-cache', 'vram', '--prefix-cache-mib', '1024']
    stop = threading.Event()
    def monitor():
        while not stop.wait(.1):
            try:
                free = gpu_free_mib()
                report['memory_samples'].append({'time': time.time(), 'free_mib': free})
                if free < 10 * 1024:
                    report['headroom_abort'] = free
                    process.terminate()
                    return
            except Exception as e:
                report['monitor_error'] = repr(e)
                process.terminate()
                return
    def read(timeout=300):
        with selectors.DefaultSelector() as sel:
            sel.register(process.stdout, selectors.EVENT_READ)
            if not sel.select(timeout):
                raise TimeoutError('native readout deadline')
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('native readout exited')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result)
        return result
    def call(body):
        process.stdin.write(json.dumps(body)+'\n'); process.stdin.flush()
        return read()
    def group(entries):
        result = call({'batch': [x['request'] for x in entries]})
        assert len(result['responses']) == len(entries)
        for r in result['responses']:
            assert 'error' not in r, r
        return result['responses']
    process = None
    monitor_thread = None
    with (out/'native.log').open('w') as log:
        try:
            process = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=log, text=True, bufsize=1)
            report['info'] = read()
            assert report['info']['ready'] and report['info']['parallel_slots'] == slots
            assert report['info']['batch_tokens'] == batch_tokens
            report['loaded'] = gpu_snapshot()
            assert gpu_free_mib() >= 10 * 1024
            monitor_thread = threading.Thread(target=monitor, daemon=True); monitor_thread.start()
            # Small canary first, followed by mixed opposite-answer states at full capacity.
            group(data['controls'][:1])
            controls = (data['controls'] * ((slots + 15)//16))[:max(16, slots)]
            for start in range(0, len(controls), slots):
                batch = controls[start:start+slots]
                for item, value in zip(batch, group(batch), strict=True):
                    p = softmax(value['logits'])[0]
                    report['controls'].append({'id': item['id'], 'expected': item['expected'], 'p': p})
                    assert (p > .5) == item['expected'], 'explicit fact control failed'
            # Warm the same shape; this pass is separate from measured rounds.
            token_counts = {}
            for start in range(0, len(items), slots):
                batch = items[start:start+slots]
                for item, value in zip(batch, group(batch), strict=True):
                    token_counts[item['id']] = value['input_tokens']
            report['token_counts'] = token_counts
            if length_order:
                items.sort(key=lambda item: token_counts[item['id']])
                for start in range(0, len(items), slots):
                    group(items[start:start+slots])
            report['order'] = [item['id'] for item in items]
            for round_index in range(rounds):
                run = {'probabilities': {}, 'exchanges': [], 'input_tokens': 0}
                if profile and round_index == 0:
                    assert call({'profile': 'start'})['ok']
                started = time.perf_counter()
                for start in range(0, len(items), slots):
                    batch = items[start:start+slots]; began = time.perf_counter()
                    results = group(batch)
                    wall = (time.perf_counter()-began)*1000
                    first = results[0]
                    run['exchanges'].append({'size': len(batch), 'wall_ms': wall,
                        **{key: first[key] for key in ['total_ms','decode_ms','decode_calls','decode_trace',
                                                      'prepare_ms','memory_ms','readout_ms','submit_ms','padding_tokens']}})
                    for item, result in zip(batch, results, strict=True):
                        run['probabilities'][item['id']] = softmax(result['logits'])[0]
                        run['input_tokens'] += result['input_tokens']
                run['wall_ms'] = (time.perf_counter()-started)*1000
                if profile and round_index == 0:
                    assert call({'profile': 'stop'})['ok']
                assert len(run['probabilities']) == len(items)
                if not limit:
                    assert run['input_tokens'] == data['expected_input_tokens'], run['input_tokens']
                run['totals'] = {key: sum(e[key] for e in run['exchanges']) for key in
                                ['decode_ms','prepare_ms','memory_ms','readout_ms','submit_ms','total_ms','decode_calls','padding_tokens']}
                run['decode_sequences'] = dict(Counter(str(d['sequences']) for e in run['exchanges'] for d in e['decode_trace']))
                run['mean_decode_tokens'] = statistics.mean(d['tokens'] for e in run['exchanges'] for d in e['decode_trace'])
                run['exchange_p50_ms'] = statistics.median(e['wall_ms'] for e in run['exchanges'])
                run['exchange_p95_ms'] = sorted(e['wall_ms'] for e in run['exchanges'])[int((len(run['exchanges'])-1)*.95)]
                run['places_per_second'] = len(items)/(run['wall_ms']/1000)
                if expected:
                    differences = [abs(p-expected[k]) for k,p in run['probabilities'].items()]
                    run['agreement'] = {'mae': statistics.mean(differences), 'max': max(differences),
                        'threshold_crossings': sum((p>.5)!=(expected[k]>.5) for k,p in run['probabilities'].items())}
                    # Report drift without treating unlabeled rankings as ground truth.
                    run['within_drift_gate'] = max(differences) <= .02
                report['runs'].append(run); save()
                print(json.dumps({'slots':slots,'batch_tokens':batch_tokens,'round':round_index+1,
                                  'wall_ms':run['wall_ms'],'totals':run['totals'],
                                  'mean_decode_tokens':run['mean_decode_tokens'],
                                  'agreement':run.get('agreement')}),flush=True)
            assert not report.get('headroom_abort') and not report.get('monitor_error')
            report['finished'] = True
        except BaseException as error:
            report['error'] = repr(error)
            raise
        finally:
            stop.set()
            if monitor_thread: monitor_thread.join(timeout=2)
            if process:
                process.stdin.close()
                try: process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
            report['after'] = gpu_snapshot()
            if report['memory_samples']:
                report['min_free_mib'] = min(s['free_mib'] for s in report['memory_samples'])
            save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['executable','model','fixture','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--projector',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--slots',type=int,choices=[4,8,16,32],required=True)
    parser.add_argument('--batch-tokens',type=int,choices=[1024,2048,4096,8192],default=1024)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--length-order',action='store_true',help='group independent prompts by measured token length')
    parser.add_argument('--profile', action='store_true', help='control CUDA capture; launch this harness under nsys')
    a=parser.parse_args()
    if not 1<=a.rounds<=10 or (a.limit is not None and not 1<=a.limit<=300):
        parser.error('rounds must be 1..10 and limit 1..300')
    measure(a.executable,a.model,a.projector,a.fixture,a.slots,a.batch_tokens,a.rounds,a.out,
            a.reference,a.profile,a.limit,a.length_order)


if __name__=='__main__':
    main()
