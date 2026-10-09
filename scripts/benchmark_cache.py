"""Exercise device snapshot ownership, pinned peers, eviction and retained allocations."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from shingi.backend import NativeReadout
from shingi.decision import prefix_for, suffix_for, softmax
from shingi.gpu import gpu_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--projector', type=Path)
    parser.add_argument('--prefix-cache', choices=('vram', 'host', 'off'), default='vram')
    parser.add_argument('--prefix-cache-mib', type=int, default=1024)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'executable_sha256': hashlib.sha256(args.executable.read_bytes()).hexdigest(),
              'policy': args.prefix_cache, 'budget_mib': args.prefix_cache_mib, 'runs': [], 'comparisons': []}
    def save():
        (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    suffixes = [{'text': suffix_for(q, [('yes', ''), ('no', '')]), 'labels': ['A', 'B']} for q in
                ['Is the parcel red?', 'Is its destination Berlin?', 'Is the parcel fragile?'] * 3]
    bodies = [{'prefix': prefix_for({'parcel': color, 'destination': destination, 'fragile': fragile,
                                    'inventory': 'Item recorded. ' * length}), 'suffixes': suffixes}
              for color, destination, fragile, length in [
                  ('red', 'Berlin', True, 0), ('blue', 'Paris', False, 300),
                  ('red', 'Rome', False, 1000), ('green', 'Berlin', True, 30),
                  ('yellow', 'Lisbon', True, 3000), ('red', 'Madrid', False, 1500)]]
    backend = None
    def call(name, body):
        start = time.perf_counter()
        result = backend._call(body, 600)
        report['runs'].append({'name': name, 'request': body, 'result': result,
                               'wall_ms': (time.perf_counter() - start) * 1000, 'memory': gpu_snapshot()})
        states = [r['cache'] for r in result.get('responses', [result]) if 'cache' in r]
        for cache in states:
            if args.prefix_cache == 'vram':
                assert cache['device_reserved_bytes'] <= args.prefix_cache_mib * 1024 ** 2, cache
                # Device snapshots retain only metadata on the host, not tensor payloads.
                assert cache['host_bytes'] < 1024 ** 2, cache
        save()
        return result
    def compare(name, actual, expected):
        for index, (a, b) in enumerate(zip(actual['results'], expected['results'], strict=True)):
            p, q = softmax(a['logits']), softmax(b['logits'])
            tvd = sum(abs(x-y) for x, y in zip(p, q))/2
            agreement = p.index(max(p)) == q.index(max(q))
            report['comparisons'].append({'name': name, 'question': index, 'tvd': tvd, 'winner': agreement})
            if tvd > .005 or not agreement:
                raise RuntimeError(f'cache ownership/agreement failed: {name} q{index}, TVD={tvd}')
    try:
        backend = NativeReadout(args.executable, args.model, args.projector, parallel=4,
                                prefix_cache=args.prefix_cache, prefix_cache_mib=args.prefix_cache_mib)
        report['info'] = backend.info
        report['loaded'] = gpu_snapshot()
        # Independent references use the same prefix boundaries and question shape;
        # cross-request caching is disabled, and each is compared with later replay.
        references = [call(f'reference-{i}', {**body, 'cache': False}) for i, body in enumerate(bodies)]
        for cycle in range(3):
            for i in [0, 1, 2, 3, 0, 4, 1, 5, 3, 2]:
                result = call(f'cycle-{cycle}-{i}', bodies[i])
                compare(f'cycle-{cycle}-{i}', result, references[i])
        # Four different prefixes remain pinned while later peers prepare and evict.
        for order in [[0, 1, 2, 3], [4, 1, 5, 0], [3, 2, 1, 0]]:
            result = call('pinned-'+str(order), {'batch': [bodies[i] for i in order]})
            for i, actual in zip(order, result['responses'], strict=True):
                compare('pinned-'+str(i), actual, references[i])
        # Rejection after successful preparation must not preserve a stale live prefix.
        invalid = {**bodies[0], 'prefix': 'oversized input. ' * 20000}
        result = call('invalid-peer', {'batch': [bodies[0], invalid, bodies[1]]})
        assert result['responses'][1]['error_kind'] == 'input'
        compare('valid-before-error', result['responses'][0], references[0])
        compare('valid-after-error', result['responses'][2], references[1])
        compare('final-replay', call('final-replay', bodies[0]), references[0])
        report['cache'] = backend.prefix_cache
        report['max_tvd'] = max(r['tvd'] for r in report['comparisons'])
        report['finished'] = True
        print(json.dumps({'finished': True, 'comparisons': len(report['comparisons']), 'max_tvd': report['max_tvd'],
                          'cache': report['cache']}), flush=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        if backend:
            backend.close()
        report['after'] = gpu_snapshot()
        save()


if __name__ == '__main__':
    main()
