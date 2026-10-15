import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare(c, out, source):
    data = pd.read_csv(source)
    required = {'architecture', 'device', 'objective', 'block', 'cost', 'cap', 'budget', 'charge', 'error', 'split', 'genotype'}
    if not required <= set(data):
        raise ValueError('Missing columns: ' + ','.join(sorted(required - set(data))))
    if (data[['cap', 'budget', 'charge']] <= 0).any().any() or (data.cost < 0).any():
        raise ValueError('Costs must be nonnegative and caps, budgets and charges positive')
    if data.duplicated(['architecture', 'device', 'objective', 'block']).any():
        raise ValueError('Run-block identities must be unique across partitions')
    data.to_csv(out / 'tape.csv', index=False)


def bounds(count, total, index, streams, c, old):
    alpha = 36 * c['delta'] / (np.pi ** 4 * index ** 2 * count ** 2 * streams)
    radius = np.sqrt(np.log(2 / alpha) / (2 * count))
    mean = total / count
    return max(old[0], mean - radius, 0), min(old[1], mean + radius, 1)


def status(active, stream_keys, lower, upper, budgets, errors, point=None):
    feasible, infeasible, dominated = set(), set(), set()
    for a in active:
        if point is None:
            if all(upper[a, k] <= budgets[k] for k in stream_keys):
                feasible.add(a)
            if any(lower[a, k] > budgets[k] for k in stream_keys):
                infeasible.add(a)
        else:
            if any(not np.isfinite(point[a, k]) for k in stream_keys):
                continue
            if all(point[a, k] <= budgets[k] for k in stream_keys):
                feasible.add(a)
            else:
                infeasible.add(a)
    viable = [a for a in active if a not in infeasible]
    for a in viable:
        for b in viable:
            if a == b or errors[a] > errors[b]:
                continue
            left = [point[a, k] if point is not None else upper[a, k] for k in stream_keys]
            right = [point[b, k] if point is not None else lower[b, k] for k in stream_keys]
            if all(x <= y for x, y in zip(left, right)) and (errors[a] < errors[b] or any(x < y for x, y in zip(left, right))):
                dominated.add(b)
    return feasible - dominated, infeasible, dominated


def graph(active, stream_keys, lower, upper, budgets, errors, excluded):
    feasibility = defaultdict(int)
    incidence = defaultdict(int)
    viable = [a for a in active if a not in excluded]
    for a in viable:
        for k in stream_keys:
            feasibility[a, k] = int(lower[a, k] <= budgets[k] < upper[a, k])
    for a in viable:
        for b in viable:
            if a == b or errors[a] > errors[b] or any(lower[a, k] > upper[b, k] for k in stream_keys):
                continue
            for k in stream_keys:
                if upper[a, k] > lower[b, k]:
                    incidence[a, k] += 1
                    incidence[b, k] += 1
    return feasibility, incidence


def run(c, out):
    tape = pd.read_csv(out / 'tape.csv')
    training = tape[tape.split == 'search']
    catalog = training.drop_duplicates('architecture').set_index('architecture')
    architecture_order = catalog.index.tolist()
    stream_keys = list(training[['device', 'objective']].drop_duplicates().itertuples(index=False, name=None))
    streams = {(a, (d, q)): g.sort_values('block') for (a, d, q), g in training.groupby(['architecture', 'device', 'objective'])}
    error = catalog.error.to_dict()
    budgets = {(d, q): float(g.budget.iloc[0] / g.cap.iloc[0]) for (d, q), g in training.groupby(['device', 'objective'])}
    results, traces = [], []
    methods = ['full', 'uniform-interval', 'graph-point', 'uniform-point', 'largest-width', 'random-active', 'no-parent-gate']
    for method in methods:
        rng = np.random.default_rng(c['seed'])
        counts, totals = defaultdict(int), defaultdict(float)
        lower, upper = defaultdict(float), defaultdict(lambda: 1.)
        active, queue, introduction = [], [], {}
        charged, steps, generation = 0., 0, 0
        conflict = False
        pending = architecture_order[:c['population']]
        next_architecture = len(pending)
        while charged < c['run_budget']:
            while pending:
                a = pending.pop(0)
                if a in introduction:
                    continue
                if any((a, k) not in streams for k in stream_keys):
                    continue
                introduction[a] = len(introduction) + 1
                active.append(a)
                queue.extend([(a, k) for k in stream_keys for _ in range(c['initial_blocks'])])
            point = {(a, k): totals[a, k] / counts[a, k] if counts[a, k] else np.nan for a in active for k in stream_keys}
            archive, infeasible, dominated = status(active, stream_keys, lower, upper, budgets, error,
                                                    point if method in ['graph-point', 'uniform-point'] else None)
            feasibility, incidence = graph(active, stream_keys, lower, upper, budgets, error, infeasible | dominated)
            total_incidence = max(1, sum(incidence.values()))
            candidates = []
            for a in active:
                for k in stream_keys:
                    key = a, k
                    priority = c['feasibility_weight'] * feasibility[key] + (1 - c['feasibility_weight']) * incidence[key] / total_incidence
                    if priority > 0 and counts[key] < len(streams[key]):
                        charge = float(streams[key].charge.iloc[counts[key]])
                        if charged + charge <= c['run_budget']:
                            candidates.append((key, priority, charge))
            if queue:
                key = queue.pop(0)
                if counts[key] >= len(streams[key]):
                    continue
                charge = float(streams[key].charge.iloc[counts[key]])
            elif candidates:
                if method in ['uniform-interval', 'uniform-point']:
                    key, _, charge = candidates[steps % len(candidates)]
                elif steps % c['forced_period'] == 0:
                    key, _, charge = candidates[(steps // c['forced_period']) % len(candidates)]
                elif method == 'random-active':
                    key, _, charge = candidates[int(rng.integers(len(candidates)))]
                else:
                    def priority(item):
                        key, incidence_score, charge = item
                        weight = 1 if method == 'largest-width' else incidence_score
                        return weight * (upper[key] - lower[key]) / ((counts[key] + 1) * charge)
                    key, _, charge = max(candidates, key=priority)
            else:
                if c['evolution'] and next_architecture < len(architecture_order):
                    pending = architecture_order[next_architecture:next_architecture + c['population']]
                    next_architecture += len(pending)
                    continue
                break
            if charged + charge > c['run_budget']:
                break
            row = streams[key].iloc[counts[key]]
            counts[key] += 1
            totals[key] += min(row.cost, row.cap) / row.cap
            lower[key], upper[key] = bounds(counts[key], totals[key], introduction[key[0]], len(stream_keys), c, (lower[key], upper[key]))
            charged += charge
            steps += 1
            if lower[key] > upper[key]:
                conflict = True
                break
            traces.append(dict(method=method, step=steps, charged=charged, architecture=key[0], device=key[1][0],
                               objective=key[1][1], lower=lower[key], upper=upper[key], count=counts[key]))
            if c['evolution'] and steps % c['generation_blocks'] == 0:
                generation += 1
                unresolved = [a for a in active if a not in archive | infeasible | dominated]
                pool = unresolved if rng.random() < c['unresolved_quota'] else list(archive)
                if method == 'no-parent-gate':
                    pool = [a for a in active if a not in infeasible | dominated]
                if not pool:
                    pool = unresolved or list(archive)
                if pool and next_architecture < len(architecture_order):
                    parent = min(rng.choice(pool, size=2, replace=True), key=lambda a: error[a])
                    genotype = np.array(json.loads(catalog.loc[parent, 'genotype']))
                    unseen = [a for a in architecture_order if a not in introduction]
                    rng.shuffle(unseen)
                    unseen.sort(key=lambda a: np.count_nonzero(np.array(json.loads(catalog.loc[a, 'genotype'])) != genotype))
                    pending = unseen[:c['population']]
                    next_architecture += len(pending)
        archive, infeasible, dominated = status(active, stream_keys, lower, upper, budgets, error,
                                                {(a, k): totals[a, k] / counts[a, k] if counts[a, k] else np.nan for a in active for k in stream_keys} if method in ['graph-point', 'uniform-point'] else None)
        chosen = min(archive, key=lambda a: (error[a], max(upper[a, k] / budgets[k] for k in stream_keys), introduction[a])) if archive and not conflict else None
        results.append(dict(method=method, selected=chosen, certified_return=chosen is not None,
                            archive=list(archive), unresolved=[a for a in active if a not in archive | infeasible | dominated],
                            excluded=list(infeasible | dominated), charged=charged, introduced=active, conflict=conflict))
    pd.DataFrame(traces).to_csv(out / 'measurements.csv', index=False)
    (out / 'search.json').write_text(json.dumps(results))


def hv2(points):
    result, height = 0., 1.
    for error, cost in sorted(points):
        if error < 1 and cost < height:
            result += (1 - error) * (height - cost)
            height = cost
    return result


def evaluate(c, out):
    tape = pd.read_csv(out / 'tape.csv')
    evaluation = tape[tape.split == 'evaluation'].copy()
    evaluation['normalized'] = np.minimum(evaluation.cost, evaluation.cap) / evaluation.cap
    truth = evaluation.groupby(['architecture', 'device', 'objective']).normalized.mean()
    limits = {key: float(g.budget.iloc[0] / g.cap.iloc[0]) for key, g in evaluation.groupby(['device', 'objective'])}
    error = tape.drop_duplicates('architecture').set_index('architecture').error.to_dict()
    feasible = {a for a in truth.index.get_level_values(0).unique() if all(truth.loc[a].get(k, np.inf) <= b for k, b in limits.items())}
    points = {a: (error[a], max(truth.loc[a][k] / b for k, b in limits.items())) for a in feasible}
    frontier = {a for a in feasible if not any(b != a and all(x <= y for x, y in zip(points[b], points[a])) and points[b] != points[a] for b in feasible)}
    rows = []
    for result in json.loads((out / 'search.json').read_text(encoding='utf-8')):
        selected = result['selected']
        relevant = frontier & set(result['introduced'])
        rows.append(dict(method=result['method'], certified_return=result['certified_return'],
                         development_accuracy=1 - error[selected] if selected is not None else np.nan,
                         heldout_violation=selected not in feasible if selected is not None else np.nan,
                         feasible_hypervolume=hv2([points[a] for a in result['archive'] if a in feasible]),
                         false_pruning=len(relevant & set(result['excluded'])) / max(1, len(relevant)),
                         unresolved_fraction=len(relevant & set(result['unresolved'])) / max(1, len(relevant)),
                         resolved_candidates_per_charge=(len(result['introduced']) - len(result['unresolved'])) / max(result['charged'], 1),
                         charged_blocks=result['charged'], observation_conflict=result['conflict']))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'measurements.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, g in d.groupby('method'):
        ax.plot(g.charged, g.upper - g.lower, alpha=.6, label=name)
    ax.set(xlabel='Charged measurement budget', ylabel='Measured stream interval width')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'intervals.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
