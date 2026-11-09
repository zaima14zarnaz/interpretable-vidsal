"""Integration tests use a temporary synthetic dataset; never the research data."""
import base64
import csv
import io
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from backend.app import Settings, create_app

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jH2QAAAAASUVORK5CYII=')


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'gallery'
    for group, count in [('7', 5), ('19', 9)]:
        for i in range(count):
            folder = root / f'group_{group}' / f'patch_{i}'
            folder.mkdir(parents=True)
            (folder / 'patch.png').write_bytes(PNG)
            (folder / 'target_frame_boxed.png').write_bytes(PNG)
    cfg = Settings(root, root / 'annotations', {'one@example.edu', 'two@example.edu'},
                   ['https://researcher.github.io'], auth_mode='console')
    app = create_app(cfg)
    return cfg, app, TestClient(app)


def login(client, capsys, email='one@example.edu'):
    assert client.post('/api/auth/request-code', json={'email': email}).status_code == 200
    code = re.search(r': (\d{6})', capsys.readouterr().out).group(1)
    response = client.post('/api/auth/verify', json={'email': email, 'code': code})
    assert response.status_code == 200, response.text
    data = response.json()
    return {'Authorization': 'Bearer ' + data['token']}, data


def start(client, headers, alias):
    r = client.post(f'/api/samples/{alias}/start', headers=headers)
    assert r.status_code == 200, r.text
    s = r.json()
    return s, {**headers, 'X-Visit-ID': s['visit_id']}


def complete_sample(client, headers, alias, body=None):
    s, h = start(client, headers, alias)
    body = body or {'description': 'Repeated horizontal lines', 'coherence': 4,
                    'outlier_mode': 'selected', 'outliers': [s['patches'][0]]}
    assert client.put(f'/api/samples/{alias}/draft', json=body, headers=h).status_code == 200
    result = client.post(f'/api/samples/{alias}/submit', json=body, headers=h)
    assert result.status_code == 200, result.text
    return result.json(), s, body


def test_complete_exports_and_idempotence(setup, capsys):
    cfg, app, client = setup
    h, data = login(client, capsys)
    assert client.get('/api/export/annotations.csv', headers=h).status_code == 409
    state, first, body = complete_sample(client, h, 'S001')
    assert state['completed_samples'] == 1
    assert client.post('/api/samples/S001/submit', headers=h, json=body).status_code == 200
    state, second, second_body = complete_sample(client, h, 'S002')
    assert state['complete']
    pid = data['participant_id']
    folder = cfg.annotation_root / f'participant_{pid}'
    metadata = json.loads((folder / 'metadata.json').read_text())
    rows = list(csv.DictReader((folder / 'annotations.csv').open()))
    assert len(rows) == 2
    assert metadata['email'] == 'one@example.edu'
    assert metadata['total_samples'] == 2 and metadata['total_patches'] == 14
    assert metadata['sample_unit'] == 'concept_group'
    assert metadata['average_time_per_sample_seconds'] == round(metadata['total_annotation_time_seconds']/2, 3)
    for row in rows:
        mapping = json.loads(row['patch_order_json'])
        assert json.loads(row['outlier_patch_ids_json']) == [mapping['P01']]
    # Exactly one row per concept even when final response is retried or two requests race.
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(lambda _: client.post('/api/samples/S002/submit', json=second_body, headers=h).status_code, range(2)))
    assert codes == [200, 200]
    assert len(list(csv.DictReader((folder / 'annotations.csv').open()))) == 2
    assert client.get('/api/export/annotations.csv', headers=h).status_code == 200
    # Missing output files are repaired from the authoritative database on the next state request.
    (folder / 'annotations.csv').unlink()
    assert client.get('/api/study', headers=h).status_code == 200
    assert (folder / 'annotations.csv').is_file()


def test_auth_and_isolation(setup, capsys):
    _, _, client = setup
    assert client.get('/api/study').status_code == 401
    assert client.get('/api/images/S001/P01/patch').status_code == 401
    assert client.post('/api/auth/request-code', json={'email':'stranger@example.edu'}).status_code == 200
    assert not capsys.readouterr().out
    h1, p1 = login(client, capsys)
    h2, p2 = login(client, capsys, 'two@example.edu')
    assert p1['participant_id'] != p2['participant_id']
    complete_sample(client, h1, 'S001')
    assert client.get('/api/study', headers=h2).json()['completed_samples'] == 0
    assert client.get('/api/export/metadata.json', headers=h2).status_code == 409
    assert client.post('/api/auth/logout', headers=h1).status_code == 200
    assert client.get('/api/study', headers=h1).status_code == 401


def test_code_attempt_limit_and_reuse(setup, capsys):
    _, _, client = setup
    email = 'one@example.edu'
    client.post('/api/auth/request-code', json={'email':email})
    code = re.search(r': (\d{6})', capsys.readouterr().out).group(1)
    wrong = '000001' if code != '000001' else '000002'
    for _ in range(5):
        assert client.post('/api/auth/verify', json={'email':email,'code':wrong}).status_code == 401
    assert client.post('/api/auth/verify', json={'email':email,'code':code}).status_code == 401
    assert client.post('/api/auth/request-code', json={'email':email}).status_code == 429


def test_resume_randomization_and_blinding(setup, capsys):
    _, _, client = setup
    h, p = login(client, capsys)
    s, lease = start(client,h,'S001')
    assert 'group_id' not in s and 'title' not in s and 'description' not in s
    body = {'description':'My own description','coherence':None,'outlier_mode':None,'outliers':[]}
    assert client.put('/api/samples/S001/draft',json=body,headers=lease).status_code == 200
    new, new_lease = start(client,h,'S001')
    assert new['patches'] == s['patches'] and new['draft'] == body
    # A stale second tab cannot overwrite the active tab's draft or submit answers.
    assert client.put('/api/samples/S001/draft',json=body,headers=lease).status_code == 409
    assert client.post('/api/samples/S001/tick',json={'visit_id':s['visit_id'],'seconds':2},headers=h).status_code == 409
    assert client.post('/api/samples/S002/start',headers=h).status_code == 409
    assert client.get('/api/images/S002/P01/patch',headers=h).status_code == 409
    assert client.get('/api/images/S001/P99/patch',headers=h).status_code == 404
    assert client.get('/api/images/S001/P01/patch',headers=h).content == PNG


def test_validation_timing_and_frame_audit(setup, capsys):
    cfg, app, client = setup
    h, p = login(client,capsys)
    s, lease = start(client,h,'S001')
    bad = {'description':'','coherence':5,'outlier_mode':'none','outliers':[]}
    assert client.post('/api/samples/S001/submit',json=bad,headers=lease).status_code == 422
    bad.update(description='Texture',outlier_mode='selected')
    assert client.post('/api/samples/S001/submit',json=bad,headers=lease).status_code == 422
    bad.update(outliers=['P99'])
    assert client.post('/api/samples/S001/submit',json=bad,headers=lease).status_code == 422
    bad.update(outliers=['P01','P01'])
    assert client.post('/api/samples/S001/submit',json=bad,headers=lease).status_code == 422
    assert client.get('/api/images/S001/P01/frame',headers=h).status_code == 200
    assert client.get('/api/images/S001/P01/frame',headers=h).status_code == 200
    with sqlite3.connect(app.state.database) as c:
        c.execute('UPDATE samples SET last_tick=? WHERE participant=?',(time.time()-4,p['participant_id']))
    assert client.post('/api/samples/S001/tick',headers=h,json={'visit_id':s['visit_id'],'seconds':4}).status_code == 200
    bad.update(outliers=['P01'])
    assert client.post('/api/samples/S001/submit',json=bad,headers=lease).status_code == 200
    complete_sample(client,h,'S002')
    rows = list(csv.DictReader(io.StringIO(client.get('/api/export/annotations.csv',headers=h).text)))
    assert json.loads(rows[0]['frame_viewed_display_ids_json']) == ['P01']
    assert 3.9 <= float(rows[0]['active_seconds']) <= 4.1


def test_manifest_and_asset_safety(setup):
    cfg, _, _ = setup
    changed = cfg.data_root / 'group_7' / 'patch_0' / 'patch.png'
    changed.write_bytes(PNG + b'changed')
    with pytest.raises(RuntimeError,match='Study images/configuration changed'):
        create_app(cfg)
    changed.unlink()
    changed.symlink_to('/etc/passwd')
    with pytest.raises(RuntimeError,match='Missing or unsafe'):
        create_app(cfg)


def test_frame_disabled_and_cors(setup, capsys, tmp_path):
    cfg, _, _ = setup
    cfg.annotation_root = tmp_path/'patch-only-annotations'
    cfg.frame_context = False
    client = TestClient(create_app(cfg))
    h,_ = login(client,capsys)
    start(client,h,'S001')
    assert client.get('/api/images/S001/P01/frame',headers=h).status_code == 403
    r = client.options('/api/study',headers={'Origin':'https://researcher.github.io',
                     'Access-Control-Request-Method':'GET','Access-Control-Request-Headers':'authorization'})
    assert r.headers['access-control-allow-origin'] == 'https://researcher.github.io'
    r = client.options('/api/study',headers={'Origin':'https://unrelated.example',
                     'Access-Control-Request-Method':'GET'})
    assert 'access-control-allow-origin' not in r.headers


def test_export_failure_retry_and_formula_protection(setup,capsys,monkeypatch):
    from backend import app as module
    _,_,client=setup
    h,_=login(client,capsys)
    complete_sample(client,h,'S001')
    s,lease=start(client,h,'S002')
    body={'description':'=unsafe spreadsheet formula','coherence':2,'outlier_mode':'cannot_determine','outliers':[]}
    original=module.atomic_write
    def fail(path,data):
        if path.name=='metadata.json':
            raise OSError('simulated disk failure')
        return original(path,data)
    monkeypatch.setattr(module,'atomic_write',fail)
    with pytest.raises(OSError,match='simulated disk failure'):
        client.post('/api/samples/S002/submit',json=body,headers=lease)
    monkeypatch.setattr(module,'atomic_write',original)
    r=client.post('/api/samples/S002/submit',json=body,headers=lease)
    assert r.status_code==200 and r.json()['complete']
    rows=list(csv.DictReader(io.StringIO(client.get('/api/export/annotations.csv',headers=h).text)))
    assert rows[1]['description'].startswith("'=")


def test_preview_mode_startup_sessions_and_exports(tmp_path):
    root = tmp_path / 'gallery'
    for group, count in [('7', 2), ('19', 3)]:
        for i in range(count):
            folder = root / f'group_{group}' / f'patch_{i}'
            folder.mkdir(parents=True)
            (folder / 'patch.png').write_bytes(PNG)
            (folder / 'target_frame_boxed.png').write_bytes(PNG)
    annotation_root = root / 'annotations'
    # No ALLOWED_EMAILS or SMTP settings required for preview.
    cfg = Settings(root, annotation_root, set(), [], auth_mode='preview')
    app = create_app(cfg)
    client = TestClient(app)
    assert cfg.annotation_root == (annotation_root / 'preview').resolve()
    public = client.get('/api/config').json()
    assert public == {'auth_mode': 'preview', 'frame_context': True, 'protocol': 'coherence-v1'}
    assert 'email' not in public and 'smtp' not in public and 'path' not in str(public).lower()
    assert client.post('/api/auth/request-code', json={'email': 'one@example.edu'}).status_code == 404
    first = client.post('/api/auth/preview-session')
    second = client.post('/api/auth/preview-session')
    assert first.status_code == 200 and second.status_code == 200
    a, b = first.json(), second.json()
    assert a['participant_id'] != b['participant_id']
    assert a['token'] != b['token']
    assert a['email'].startswith('preview+') and a['is_test_data'] is True
    h1 = {'Authorization': 'Bearer ' + a['token']}
    h2 = {'Authorization': 'Bearer ' + b['token']}
    assert client.get('/api/study', headers=h1).json()['participant_id'] == a['participant_id']
    assert client.get('/api/study', headers=h2).json()['completed_samples'] == 0
    complete_sample(client, h1, 'S001')
    state, _, _ = complete_sample(client, h1, 'S002')
    assert state['complete'] and state['is_test_data'] is True
    folder = cfg.annotation_root / f"participant_{a['participant_id']}"
    assert folder.is_dir()
    assert not (annotation_root / f"participant_{a['participant_id']}").exists()
    metadata = json.loads((folder / 'metadata.json').read_text())
    assert metadata['is_test_data'] is True
    assert metadata['auth_mode'] == 'preview'
    assert list(csv.DictReader((folder / 'annotations.csv').open()))


def test_preview_endpoint_disabled_outside_preview(setup):
    _, _, client = setup
    assert client.get('/api/config').json()['auth_mode'] == 'console'
    assert client.post('/api/auth/preview-session').status_code == 404
    assert client.post('/api/auth/email-session', json={'email': 'one@example.edu'}).status_code == 404


def _gallery(tmp_path, groups=(('7', 2), ('19', 3))):
    root = tmp_path / 'gallery'
    for group, count in groups:
        for i in range(count):
            folder = root / f'group_{group}' / f'patch_{i}'
            folder.mkdir(parents=True)
            (folder / 'patch.png').write_bytes(PNG)
            (folder / 'target_frame_boxed.png').write_bytes(PNG)
    return root


def test_email_only_entry_resume_and_isolation(tmp_path):
    root = _gallery(tmp_path)
    cfg = Settings(root, root / 'annotations', set(), [], auth_mode='email_only')
    client = TestClient(create_app(cfg))
    assert client.get('/api/config').json()['auth_mode'] == 'email_only'
    assert client.post('/api/auth/request-code', json={'email': 'a@example.edu'}).status_code == 404
    assert client.post('/api/auth/preview-session').status_code == 404

    # Any nonempty entry is accepted after trim/lowercase; no domain/invitation checks.
    odd = client.post('/api/auth/email-session', json={'email': '  Mixed.Case@Example.EDU '})
    assert odd.status_code == 200, odd.text
    data = odd.json()
    assert data['email'] == 'mixed.case@example.edu'
    assert data['email_verified'] is False
    assert data['auth_mode'] == 'email_only'
    assert data['is_test_data'] is False
    h1 = {'Authorization': 'Bearer ' + data['token']}

    # Returning with the same normalized email resumes the same participant.
    again = client.post('/api/auth/email-session', json={'email': 'mixed.case@example.edu'}).json()
    assert again['participant_id'] == data['participant_id']
    assert again['token'] != data['token']

    complete_sample(client, h1, 'S001')
    resumed = client.get('/api/study', headers={'Authorization': 'Bearer ' + again['token']}).json()
    assert resumed['completed_samples'] == 1
    assert resumed['participant_id'] == data['participant_id']

    other = client.post('/api/auth/email-session', json={'email': 'other@example.edu'}).json()
    assert other['participant_id'] != data['participant_id']
    h2 = {'Authorization': 'Bearer ' + other['token']}
    assert client.get('/api/study', headers=h2).json()['completed_samples'] == 0

    state, _, _ = complete_sample(client, h1, 'S002')
    assert state['complete']
    folder = cfg.annotation_root / f"participant_{data['participant_id']}"
    assert folder.is_dir()
    assert 'preview' not in folder.parts[-2:]
    metadata = json.loads((folder / 'metadata.json').read_text())
    assert metadata['email'] == 'mixed.case@example.edu'
    assert metadata['email_verified'] is False
    assert metadata['auth_mode'] == 'email_only'
    assert metadata['is_test_data'] is False


def test_email_only_startup_skips_allowlist_and_smtp(tmp_path):
    root = _gallery(tmp_path)
    cfg = Settings(root, root / 'annotations', set(), [], auth_mode='email_only',
                   smtp_host='', smtp_from='')
    app = create_app(cfg)
    assert app.state.settings.auth_mode == 'email_only'
    assert app.state.settings.annotation_root == (root / 'annotations').resolve()
