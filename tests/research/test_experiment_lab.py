import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab = load('experiment_lab')
compare = load('compare_benchmarks')
NO_GPU = {'available': False, 'reason': 'test host has no GPU', 'devices': []}
GPU = {'available': True, 'devices': [{'index': '0', 'uuid': 'GPU-abc'}]}
ITEM = {'id': 'fixture', 'requires_gpu': False, 'gpus_min': 0}


def recipe(tmp_path, code):
    return {'argv': [sys.executable, '-c', code], 'cwd': str(tmp_path), 'model_revision': 'test-only'}


def test_catalog_is_valid_and_has_explicit_rejection_criteria():
    catalog = lab.validate_catalog(json.loads((ROOT / 'experiments/gpu-candidates.json').read_text()))
    assert len(catalog['experiments']) == 24
    assert all(e['reject_if'] and e['hypothesis'] for e in catalog['experiments'])


def test_duplicate_catalog_id_rejected():
    item = dict(ITEM, hypothesis='test', reject_if='fails')
    with pytest.raises(ValueError, match='duplicate'):
        lab.validate_catalog({'experiments': [item, item]})


def test_gpu_aliases_cannot_count_as_two_cards():
    assert lab.normalize_gpus(['0'], GPU) == ['GPU-abc']
    with pytest.raises(ValueError, match='distinct'):
        lab.normalize_gpus(['0', 'GPU-abc'], GPU)
    with pytest.raises(ValueError, match='visible'):
        lab.normalize_gpus(['1'], GPU)


def test_missing_gpu_is_blocked_without_starting_command(tmp_path):
    result = lab.run_one(dict(ITEM, requires_gpu=True, gpus_min=1), recipe(tmp_path, 'raise AssertionError()'), tmp_path, [], NO_GPU, 1)
    assert result['status'] == 'blocked'
    assert result['reason'] == NO_GPU['reason']


def test_gpu_count_and_missing_recipe_are_blocked(tmp_path):
    item = dict(ITEM, requires_gpu=True, gpus_min=2)
    assert lab.run_one(item, recipe(tmp_path, ''), tmp_path, ['GPU-abc'], GPU, 1)['status'] == 'blocked'
    assert lab.run_one(item, None, tmp_path, [], NO_GPU, 1)['status'] == 'blocked'


def test_success_preserves_logs_but_never_passes_quality(tmp_path):
    result = lab.run_one(ITEM, recipe(tmp_path, 'import os; print(os.environ["CUDA_VISIBLE_DEVICES"]); print(os.environ["EXPERIMENT_OUTPUT_DIR"])'), tmp_path, ['GPU-abc'], GPU, 1)
    assert result['status'] == 'completed'
    assert result['quality_status'] == 'unreviewed'
    folder = next(tmp_path.glob('fixture-*'))
    assert 'GPU-abc' in (folder / 'stdout.log').read_text()
    assert json.loads((folder / 'run.json').read_text()) == result


def test_invalid_recipe_is_recorded_instead_of_raising(tmp_path):
    result = lab.run_one(ITEM, {'argv': 'echo unsafe shell', 'cwd': str(tmp_path)}, tmp_path, [], NO_GPU, 1)
    assert result['status'] == 'failed'
    assert 'argv' in result['reason']


def test_timeout_reaps_local_process(tmp_path):
    code = 'import os,time; from pathlib import Path; Path("pid").write_text(str(os.getpid())); time.sleep(30)'
    result = lab.run_one(ITEM, recipe(tmp_path, code), tmp_path, [], NO_GPU, .3)
    assert result['status'] == 'timed_out'
    pid = int((tmp_path / 'pid').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cli_continues_after_failure_and_returns_nonzero(tmp_path):
    catalog = {'experiments': [dict(ITEM, id=name, hypothesis='test', reject_if='failure') for name in ('bad', 'good')]}
    recipes = {name: recipe(tmp_path, code) for name, code in [('bad', 'raise SystemExit(3)'), ('good', 'print("success")')]}
    (tmp_path / 'catalog.json').write_text(json.dumps(catalog))
    (tmp_path / 'recipes.json').write_text(json.dumps(recipes))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/experiment_lab.py'), '--catalog', str(tmp_path / 'catalog.json'), '--recipes', str(tmp_path / 'recipes.json'), '--output', str(tmp_path / 'results'), '--execute'], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert [r['status'] for r in json.loads(result.stdout)] == ['failed', 'completed']
    assert len(list((tmp_path / 'results').glob('*/run.json'))) == 2


def row(**kwargs):
    return dict(tag='baseline', mode='fast', job={'status': 'completed'}, wall_seconds=2,
                fixture_sha256='fixture', target_language='es', phase='warm-models', hardware_profile='shared-2',
                gpu='GPU-abc sm120 driver580', commit='abc', tracked_diff_sha256='diff', untracked_source_sha256='source', **kwargs)


@pytest.mark.parametrize('field,value', [('fixture_sha256', 'other'), ('target_language', 'ja'), ('phase', 'cold'), ('hardware_profile', 'exclusive-8'), ('gpu', 'different GPU'), ('commit', 'def'), ('tracked_diff_sha256', 'new'), ('untracked_source_sha256', 'new')])
def test_incompatible_runs_are_not_pooled(field, value):
    a, b = row(), row()
    b[field] = value
    assert len(compare.summarize([a, b])) == 2


def test_candidates_share_cohort_only_with_matching_workload_and_hardware():
    a, b = row(), row()
    b.update(tag='candidate', configuration={'compile': True}, commit='other')
    summary = compare.summarize([a, b])
    assert len(summary) == 2
    assert summary[0]['cohort_sha256'] == summary[1]['cohort_sha256']


def test_failures_missing_reviews_and_small_sample_are_visible():
    a, b = row(), row()
    b['job'] = {'status': 'failed'}
    b['wall_seconds'] = 100
    result = compare.summarize([a, b])[0]
    assert result['failed_or_cancelled'] == 1
    assert result['median_seconds'] == 2
    assert result['p95_seconds'] is None
    assert result['quality_reviewed'] == 0
    assert result['provenance_complete']


def test_empirical_p95_requires_twenty_completed_runs():
    rows = [dict(row(), wall_seconds=i) for i in range(1, 21)]
    assert compare.summarize(rows)[0]['p95_seconds'] == 19


def test_legacy_rows_remain_readable_but_unqualified():
    result = compare.summarize([{'mode': 'fast', 'wall_seconds': 1, 'job': {'status': 'completed'}}])[0]
    assert not result['provenance_complete']
    assert result['quality_status'] == 'unreviewed or incomplete'


def test_frame_probe_fails_closed_without_cuda(monkeypatch, tmp_path):
    from types import SimpleNamespace
    module = load('probe_gpu_frames')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    result = module.probe(tmp_path / 'unused.mp4')
    assert result['status'] == 'blocked'
    assert result['quality_status'] == 'unreviewed'


def test_frame_probe_validates_bounds_before_importing_gpu_libraries(tmp_path):
    module = load('probe_gpu_frames')
    with pytest.raises(ValueError):
        module.probe(tmp_path / 'unused.mp4', batch_size=100)


def test_untracked_source_content_changes_fingerprint(monkeypatch, tmp_path):
    module = load('benchmark_modes')
    source = tmp_path / 'new.py'
    source.write_text('one')
    (tmp_path / 'weights.bin').write_bytes(b'not source')
    monkeypatch.setattr(module, 'command', lambda *args: str(tmp_path) if '--show-toplevel' in args else 'new.py\0weights.bin\0')
    first = module.untracked_source_manifest()
    source.write_text('two')
    second = module.untracked_source_manifest()
    assert list(first) == ['new.py']
    assert first != second


def test_benchmark_records_rejected_submissions(tmp_path, monkeypatch):
    import httpx
    from types import SimpleNamespace
    module = load('benchmark_modes')
    fixture = tmp_path / 'clip.mp4'
    fixture.write_bytes(b'fixture')
    statuses = iter([413, 503])
    def handle(request):
        if request.url.path == '/health':
            return httpx.Response(200, json={'status': 'ok'})
        assert request.method == 'POST' and request.url.path == '/jobs'
        return httpx.Response(next(statuses), json={'detail': 'private upstream response'})
    client_class = httpx.Client
    monkeypatch.setattr(module.httpx, 'Client', lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr(module, 'command', lambda *args: None)
    monkeypatch.setattr(module, 'probe', lambda *args: None)
    args = SimpleNamespace(output=tmp_path / 'results', source_provenance=None,
        fixture=fixture, tag='test', target='es', phase='cold', hardware_profile='test',
        api='http://test', config=[], modes=['fast'], repeats=2, options='{}', timeout=1)
    assert module.run(args) == 1
    content = (args.output / 'results.jsonl').read_text()
    records = [json.loads(line) for line in content.splitlines()]
    assert [row['submission_http_status'] for row in records] == [413, 503]
    assert 'private upstream response' not in content
    assert compare.summarize(records)[0]['failed_or_cancelled'] == 2
    assert compare.summarize(records)[0]['completed'] == 0
