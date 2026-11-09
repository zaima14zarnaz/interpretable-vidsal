"""Unlabeled concept coherence study. Run on the server that owns DATA_ROOT."""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import logging
import os
import random
import re
import secrets
import smtplib
import sqlite3
import ssl
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v5')
PROTOCOL = 'coherence-v1'
log = logging.getLogger('survey')


@dataclass
class Settings:
    data_root: Path
    annotation_root: Path
    allowed_emails: set[str] = field(default_factory=set)
    origins: list[str] = field(default_factory=list)
    group_ids: set[str] = field(default_factory=set)
    auth_mode: str = 'smtp'
    smtp_host: str = ''
    smtp_port: int = 587
    smtp_user: str = ''
    smtp_password: str = ''
    smtp_from: str = ''
    smtp_security: str = 'starttls'
    frame_context: bool = True

    @classmethod
    def from_env(cls):
        load_dotenv(REPO / '.env')
        root = Path(os.getenv('DATA_ROOT', str(DEFAULT_ROOT)))
        split = lambda name: {x.strip() for x in os.getenv(name, '').split(',') if x.strip()}
        return cls(root, Path(os.getenv('ANNOTATION_ROOT', str(root / 'annotations'))),
                   {x.lower() for x in split('ALLOWED_EMAILS')}, sorted(split('ALLOWED_ORIGINS')),
                   split('GROUP_IDS'), os.getenv('AUTH_MODE', 'smtp'),
                   os.getenv('SMTP_HOST', ''), int(os.getenv('SMTP_PORT', '587')),
                   os.getenv('SMTP_USER', ''), os.getenv('SMTP_PASSWORD', ''),
                   os.getenv('SMTP_FROM', ''), os.getenv('SMTP_SECURITY', 'starttls'),
                   os.getenv('ALLOW_FRAME_CONTEXT', 'true').lower() == 'true')


def utc(ts=None):
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_write(path: Path, data: str):
    """Write on the same filesystem, fsync, then publish with atomic replacement."""
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class EmailInput(BaseModel):
    email: str = Field(min_length=3, max_length=254)

    @field_validator('email')
    @classmethod
    def normalize(cls, value):
        value = value.strip().lower()
        if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value):
            raise ValueError('Enter a valid email address.')
        return value


class VerifyInput(EmailInput):
    code: str = Field(pattern=r'^\d{6}$')


class Answer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    description: str = Field(default='', max_length=2000)
    coherence: int | None = Field(default=None, ge=1, le=5, strict=True)
    outlier_mode: Literal['selected', 'none', 'cannot_determine'] | None = None
    outliers: list[str] = Field(default_factory=list, max_length=100)


class Tick(BaseModel):
    visit_id: str = Field(max_length=100)
    seconds: float = Field(ge=0, le=15, allow_inf_nan=False)


def create_app(settings: Settings | None = None):
    cfg = settings or Settings.from_env()
    cfg.data_root = cfg.data_root.resolve()
    cfg.annotation_root = cfg.annotation_root.resolve()
    if not cfg.data_root.is_dir():
        raise RuntimeError(f'Dataset directory does not exist: {cfg.data_root}')
    if not cfg.allowed_emails:
        raise RuntimeError('Set ALLOWED_EMAILS to the invited participant emails before starting.')
    if cfg.auth_mode not in {'smtp', 'console'}:
        raise RuntimeError('AUTH_MODE must be smtp or console.')
    if cfg.auth_mode == 'smtp' and (not cfg.smtp_host or not cfg.smtp_from):
        raise RuntimeError('Configure SMTP_HOST and SMTP_FROM, or use console mode for local testing.')
    if cfg.smtp_security not in {'starttls', 'ssl'}:
        raise RuntimeError('SMTP_SECURITY must be starttls or ssl.')
    cfg.annotation_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = cfg.annotation_root / '.study_state'
    state.mkdir(exist_ok=True, mode=0o700)
    secret_file = state / 'secret.key'
    try:
        with secret_file.open('x', encoding='utf-8') as f:
            os.chmod(secret_file, 0o600)
            f.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    secret = secret_file.read_text().strip().encode()

    # Only exact asset filenames are indexed. Titles and other metadata never enter the API.
    manifest = []
    natural = lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name)]
    for group in sorted(cfg.data_root.glob('group_*'), key=natural):
        if not group.is_dir() or (cfg.group_ids and group.name[6:] not in cfg.group_ids):
            continue
        patches = []
        for patch in sorted(group.glob('patch_*'), key=natural):
            if not patch.is_dir():
                continue
            files = {}
            for kind, filename in [('patch', 'patch.png'), ('frame', 'target_frame_boxed.png')]:
                p = (patch / filename).resolve()
                if not p.is_relative_to(cfg.data_root) or not p.is_file():
                    raise RuntimeError(f'Missing or unsafe study image: {patch / filename}')
                files[kind] = {'path': str(p.relative_to(cfg.data_root)),
                               'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            patches.append({'id': patch.name[6:], **files})
        if not patches:
            raise RuntimeError(f'No patch_* directories found in {group}')
        manifest.append({'id': group.name[6:], 'patches': patches})
    if not manifest:
        raise RuntimeError('No groups found. Check DATA_ROOT and GROUP_IDS.')
    if cfg.group_ids - {g['id'] for g in manifest}:
        raise RuntimeError('GROUP_IDS includes a group that was not found.')
    frozen = {'protocol': PROTOCOL, 'allow_frame_context': cfg.frame_context, 'groups': manifest}
    manifest_text = json.dumps(frozen, sort_keys=True)
    manifest_sha = digest(manifest_text)
    manifest_file = state / 'study_manifest.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != frozen:
        raise RuntimeError('Study images/configuration changed. Use a NEW ANNOTATION_ROOT for a new study.')
    atomic_write(manifest_file, json.dumps(frozen, indent=2))
    group_index = {g['id']: g for g in manifest}
    database = state / 'study.sqlite3'

    @contextmanager
    def db():
        conn = sqlite3.connect(database, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            # Serializes mutations and export recovery; intended for a small single-host study.
            conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS participants (
          id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, created REAL NOT NULL,
          started REAL, completed REAL, assignment TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS codes (
          email TEXT PRIMARY KEY, hash TEXT NOT NULL, expires REAL NOT NULL,
          issued REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS sessions (
          hash TEXT PRIMARY KEY, participant TEXT NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS auth_limits (
          key TEXT NOT NULL, stamp REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS auth_limits_idx ON auth_limits(key,stamp);
        CREATE TABLE IF NOT EXISTS samples (
          participant TEXT NOT NULL, position INTEGER NOT NULL, draft TEXT NOT NULL DEFAULT '{}',
          answer TEXT, started REAL, submitted REAL, active REAL NOT NULL DEFAULT 0,
          visit TEXT, last_tick REAL, frame_views TEXT NOT NULL DEFAULT '[]',
          PRIMARY KEY(participant,position));
        ''')
    os.chmod(database, 0o600)
    app = FastAPI(title='Visual Pattern Study', docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = cfg
    app.state.database = database
    if cfg.origins:
        if '*' in cfg.origins:
            raise RuntimeError('ALLOWED_ORIGINS must contain exact HTTPS frontend origins, not *.')
        app.add_middleware(CORSMiddleware, allow_origins=cfg.origins,
                           allow_methods=['GET', 'POST', 'PUT'], allow_headers=['Authorization', 'Content-Type', 'X-Visit-ID'])

    @app.middleware('http')
    async def headers(request: Request, call_next):
        if int(request.headers.get('content-length', '0') or 0) > 16384:
            return Response(status_code=413)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    def identity(authorization: str = Header(default='')):
        if not authorization.startswith('Bearer '):
            raise HTTPException(401, 'Sign in to continue.')
        with db() as c:
            row = c.execute('SELECT p.* FROM sessions s JOIN participants p ON p.id=s.participant '
                            'WHERE s.hash=? AND s.expires>?',
                            (digest(authorization[7:]), time.time())).fetchone()
        if not row or row['email'] not in cfg.allowed_emails:
            raise HTTPException(401, 'Your session has expired. Sign in again to resume.')
        return dict(row)

    def get_participant(c, pid):
        return dict(c.execute('SELECT * FROM participants WHERE id=?', (pid,)).fetchone())

    def position(alias):
        if not re.fullmatch(r'S\d{3,}', alias):
            raise HTTPException(404, 'Sample not found.')
        return int(alias[1:]) - 1

    def require_current(c, pid, pos):
        count = c.execute('SELECT COUNT(*) FROM samples WHERE participant=? AND answer IS NOT NULL', (pid,)).fetchone()[0]
        if pos != count or pos >= len(manifest):
            raise HTTPException(409, 'This sample is locked. Reload to resume the current sample.')

    def require_visit(c, pid, pos, visit):
        row = c.execute('SELECT visit FROM samples WHERE participant=? AND position=?', (pid, pos)).fetchone()
        if not row or not visit or row['visit'] != visit:
            raise HTTPException(409, 'This survey is open in another tab. Reload this tab to continue.')

    def assignment_patch(p, pos, patch_alias):
        assigned = json.loads(p['assignment'])
        if not 0 <= pos < len(assigned) or not re.fullmatch(r'P\d{2,}', patch_alias):
            raise HTTPException(404, 'Patch not found.')
        item = assigned[pos]
        idx = int(patch_alias[1:]) - 1
        if not 0 <= idx < len(item['patches']):
            raise HTTPException(404, 'Patch not found.')
        patch = next(x for x in group_index[item['group']]['patches'] if x['id'] == item['patches'][idx])
        return patch

    def export(c, p):
        """SQLite is authoritative; retrying completion repairs interrupted file export."""
        rows = c.execute('SELECT * FROM samples WHERE participant=? AND answer IS NOT NULL ORDER BY position',
                         (p['id'],)).fetchall()
        if len(rows) != len(manifest) or p['completed'] is None:
            raise HTTPException(409, 'Complete every sample before exporting.')
        folder = cfg.annotation_root / ('participant_' + p['id'])
        folder.mkdir(exist_ok=True, mode=0o700)
        assignment = json.loads(p['assignment'])
        output = io.StringIO(newline='')
        fields = ['participant_id', 'study_sha256', 'sample_order', 'sample_alias', 'group_id',
                  'patch_order_json', 'description', 'coherence', 'outlier_mode', 'outlier_patch_ids_json',
                  'outlier_display_ids_json', 'frame_viewed_display_ids_json', 'started_at_utc',
                  'submitted_at_utc', 'active_seconds', 'elapsed_seconds']
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        active_total = 0
        sample_elapsed_total = 0
        for row in rows:
            a = json.loads(row['answer'])
            item = assignment[row['position']]
            mapping = {f'P{i+1:02d}': x for i, x in enumerate(item['patches'])}
            active_total += row['active']
            sample_elapsed_total += row['submitted'] - row['started']
            # Prevent spreadsheet formula execution in researcher-supplied free text exports.
            description = a['description']
            if description.lstrip().startswith(('=', '+', '-', '@')):
                description = "'" + description
            writer.writerow(dict(participant_id=p['id'], study_sha256=manifest_sha,
                sample_order=row['position']+1, sample_alias=f"S{row['position']+1:03d}", group_id=item['group'],
                patch_order_json=json.dumps(mapping), description=description, coherence=a['coherence'],
                outlier_mode=a['outlier_mode'], outlier_patch_ids_json=json.dumps([mapping[x] for x in a['outliers']]),
                outlier_display_ids_json=json.dumps(a['outliers']), frame_viewed_display_ids_json=row['frame_views'],
                started_at_utc=utc(row['started']), submitted_at_utc=utc(row['submitted']),
                active_seconds=round(row['active'], 3), elapsed_seconds=round(row['submitted']-row['started'], 3)))
        csv_text = output.getvalue()
        metadata = dict(participant_id=p['id'], email=p['email'], protocol=PROTOCOL,
            study_sha256=manifest_sha, status='complete', sample_unit='concept_group',
            total_samples=len(rows), total_patches=sum(len(x['patches']) for x in assignment),
            registered_at_utc=utc(p['created']), started_at_utc=utc(p['started']), completed_at_utc=utc(p['completed']),
            total_annotation_time_seconds=round(active_total, 3),
            average_time_per_sample_seconds=round(active_total / len(rows), 3),
            total_elapsed_time_seconds=round(p['completed']-p['started'], 3),
            average_elapsed_time_per_sample_seconds=round(sample_elapsed_total / len(rows), 3),
            timing_definition='Foreground time reported at most every 5s; server caps each interval at 15s and wall time. '
                              'Includes reading and thinking in a focused tab. It is not a measure of attention. '
                              'Last unsent interval may be lost on abrupt browser termination.',
            elapsed_time_definition='Wall time from first sample start to final submission, including breaks.',
            frame_context_enabled=cfg.frame_context, randomized_assignment=assignment,
            annotations_csv_sha256=hashlib.sha256(csv_text.encode()).hexdigest())
        atomic_write(folder / 'annotations.csv', csv_text)
        # Metadata last: its CSV checksum can detect a partial export after a crash.
        atomic_write(folder / 'metadata.json', json.dumps(metadata, indent=2))
        return metadata

    def study_state(c, p):
        count = c.execute('SELECT COUNT(*) FROM samples WHERE participant=? AND answer IS NOT NULL', (p['id'],)).fetchone()[0]
        result = dict(participant_id=p['id'], email=p['email'], total_samples=len(manifest),
                      completed_samples=count, complete=count == len(manifest), frame_context=cfg.frame_context)
        if result['complete']:
            result['metadata'] = export(c, p)
        return result

    @app.get('/api/health')
    def health():
        return {'ok': True}

    @app.post('/api/auth/request-code')
    def request_code(body: EmailInput, request: Request):
        email = body.email
        now = time.time()
        code = f'{secrets.randbelow(1000000):06d}'
        # Proxy IP headers are deliberately not trusted by this application.
        client_ip = request.client.host if request.client else 'unknown'
        ip_key = 'ip:' + hmac.new(secret, client_ip.encode(), hashlib.sha256).hexdigest()
        with db() as c:
            c.execute('DELETE FROM auth_limits WHERE stamp<?', (now-3600,))
            total = c.execute('SELECT COUNT(*) FROM auth_limits WHERE key=?', (ip_key,)).fetchone()[0]
            if total >= 30:
                raise HTTPException(429, 'Too many code requests. Try again later.')
            c.execute('INSERT INTO auth_limits VALUES (?,?)', (ip_key, now))
            old = c.execute('SELECT issued FROM codes WHERE email=?', (email,)).fetchone()
            if old and now-old['issued'] < 60:
                raise HTTPException(429, 'Wait 60 seconds before requesting another code.')
            if email not in cfg.allowed_emails:
                return {'message': 'If this email is invited, a sign-in code has been sent.'}
            email_key = 'email:' + digest(email)
            if c.execute('SELECT COUNT(*) FROM auth_limits WHERE key=?', (email_key,)).fetchone()[0] >= 10:
                raise HTTPException(429, 'Too many code requests. Try again later.')
            c.execute('INSERT INTO auth_limits VALUES (?,?)', (email_key, now))
            code_hash = hmac.new(secret, (email + ':' + code).encode(), hashlib.sha256).hexdigest()
            c.execute('INSERT OR REPLACE INTO codes VALUES (?,?,?,?,0)', (email, code_hash, now+600, now))
        try:
            if cfg.auth_mode == 'console':
                # Explicit development mode only; never use this with real participants.
                print(f'[LOCAL TEST ONLY] {email}: {code}', flush=True)
            else:
                msg = EmailMessage()
                msg['From'], msg['To'], msg['Subject'] = cfg.smtp_from, email, 'Your Visual Pattern Study sign-in code'
                msg.set_content(f'Your sign-in code is {code}. It expires in 10 minutes.\n'
                                'If you did not request this code, ignore this email.')
                smtp_class = smtplib.SMTP_SSL if cfg.smtp_security == 'ssl' else smtplib.SMTP
                kwargs = {'context': ssl.create_default_context()} if cfg.smtp_security == 'ssl' else {}
                with smtp_class(cfg.smtp_host, cfg.smtp_port, timeout=15, **kwargs) as server:
                    if cfg.smtp_security == 'starttls':
                        server.starttls(context=ssl.create_default_context())
                    if cfg.smtp_user:
                        server.login(cfg.smtp_user, cfg.smtp_password)
                    server.send_message(msg)
        except (OSError, smtplib.SMTPException):
            log.error('Email delivery failed; check SMTP settings.')
            raise HTTPException(503, 'The sign-in email could not be sent. Contact the study organizer.')
        return {'message': 'If this email is invited, a sign-in code has been sent.'}

    @app.post('/api/auth/verify')
    def verify(body: VerifyInput):
        now = time.time()
        failure = False
        with db() as c:
            row = c.execute('SELECT * FROM codes WHERE email=?', (body.email,)).fetchone()
            expected = hmac.new(secret, (body.email + ':' + body.code).encode(), hashlib.sha256).hexdigest()
            if body.email not in cfg.allowed_emails or not row or row['expires'] < now or row['attempts'] >= 5:
                failure = True
            elif not hmac.compare_digest(row['hash'], expected):
                c.execute('UPDATE codes SET attempts=attempts+1 WHERE email=?', (body.email,))
                failure = True
            else:
                c.execute('DELETE FROM codes WHERE email=?', (body.email,))
                p = c.execute('SELECT * FROM participants WHERE email=?', (body.email,)).fetchone()
                if not p:
                    assignment = [{'group': g['id'], 'patches': [x['id'] for x in g['patches']]} for g in manifest]
                    rng = random.SystemRandom()
                    rng.shuffle(assignment)
                    for item in assignment:
                        rng.shuffle(item['patches'])
                    pid = secrets.token_hex(12)
                    c.execute('INSERT INTO participants VALUES (?,?,?,?,?,?)',
                              (pid, body.email, now, None, None, json.dumps(assignment)))
                    p = c.execute('SELECT * FROM participants WHERE id=?', (pid,)).fetchone()
                token = secrets.token_urlsafe(32)
                c.execute('DELETE FROM sessions WHERE expires<?', (now,))
                c.execute('INSERT INTO sessions VALUES (?,?,?)', (digest(token), p['id'], now+86400))
                result = {'token': token, **study_state(c, dict(p))}
        # Commit failed-attempt counters before returning an authentication error.
        if failure:
            raise HTTPException(401, 'Invalid or expired code. Request a new code if needed.')
        return result

    @app.post('/api/auth/logout')
    def logout(authorization: str = Header(default='')):
        with db() as c:
            c.execute('DELETE FROM sessions WHERE hash=?', (digest(authorization.removeprefix('Bearer ')),))
        return {'ok': True}

    @app.get('/api/study')
    def study(p=Depends(identity)):
        with db() as c:
            return study_state(c, get_participant(c, p['id']))

    @app.post('/api/samples/{alias}/start')
    def start(alias: str, p=Depends(identity)):
        pos, now = position(alias), time.time()
        with db() as c:
            require_current(c, p['id'], pos)
            c.execute('UPDATE participants SET started=COALESCE(started,?) WHERE id=?', (now, p['id']))
            visit = secrets.token_urlsafe(20)
            c.execute('INSERT OR IGNORE INTO samples (participant,position,started) VALUES (?,?,?)', (p['id'], pos, now))
            c.execute('UPDATE samples SET visit=?,last_tick=? WHERE participant=? AND position=?', (visit, now, p['id'], pos))
            row = c.execute('SELECT * FROM samples WHERE participant=? AND position=?', (p['id'], pos)).fetchone()
            assigned = json.loads(p['assignment'])[pos]
            return {'sample_alias': alias, 'visit_id': visit, 'draft': json.loads(row['draft']),
                    'patches': [f'P{i+1:02d}' for i in range(len(assigned['patches']))]}

    @app.post('/api/samples/{alias}/tick')
    def tick(alias: str, body: Tick, p=Depends(identity)):
        pos, now = position(alias), time.time()
        with db() as c:
            require_current(c, p['id'], pos)
            row = c.execute('SELECT * FROM samples WHERE participant=? AND position=?', (p['id'], pos)).fetchone()
            if not row or row['visit'] != body.visit_id:
                raise HTTPException(409, 'This survey is open in another tab. Reload this tab to continue.')
            delta = max(0, min(body.seconds, now-row['last_tick'], 15))
            c.execute('UPDATE samples SET active=active+?,last_tick=? WHERE participant=? AND position=?',
                      (delta, now, p['id'], pos))
        return {'ok': True}

    def validate_answer(body, p, pos, final):
        assigned = json.loads(p['assignment'])[pos]
        valid = {f'P{i+1:02d}' for i in range(len(assigned['patches']))}
        if len(set(body.outliers)) != len(body.outliers) or not set(body.outliers) <= valid:
            raise HTTPException(422, 'Choose valid patch IDs without duplicates.')
        if final:
            if not body.description.strip() or body.coherence is None or body.outlier_mode is None:
                raise HTTPException(422, 'Complete all three questions before continuing.')
            if (body.outlier_mode == 'selected') != bool(body.outliers):
                raise HTTPException(422, 'Select one or more outlier patches, or choose None / Cannot determine.')

    @app.put('/api/samples/{alias}/draft')
    def draft(alias: str, body: Answer, x_visit_id: str = Header(default=''), p=Depends(identity)):
        pos = position(alias)
        with db() as c:
            require_current(c, p['id'], pos)
            require_visit(c, p['id'], pos, x_visit_id)
            validate_answer(body, p, pos, False)
            if not c.execute('SELECT 1 FROM samples WHERE participant=? AND position=?', (p['id'], pos)).fetchone():
                raise HTTPException(409, 'Start this sample first.')
            c.execute('UPDATE samples SET draft=? WHERE participant=? AND position=?',
                      (body.model_dump_json(), p['id'], pos))
        return {'saved': True}

    @app.post('/api/samples/{alias}/submit')
    def submit(alias: str, body: Answer, x_visit_id: str = Header(default=''), p=Depends(identity)):
        pos, now = position(alias), time.time()
        with db() as c:
            existing = c.execute('SELECT answer FROM samples WHERE participant=? AND position=?', (p['id'], pos)).fetchone()
            if existing and existing['answer'] is not None:
                if json.loads(existing['answer']) != body.model_dump():
                    raise HTTPException(409, 'Submitted answers are locked.')
                return study_state(c, get_participant(c, p['id']))
            require_current(c, p['id'], pos)
            require_visit(c, p['id'], pos, x_visit_id)
            validate_answer(body, p, pos, True)
            if not existing:
                raise HTTPException(409, 'Start this sample first.')
            c.execute('UPDATE samples SET answer=?,draft=?,submitted=?,visit=NULL WHERE participant=? AND position=?',
                      (body.model_dump_json(), body.model_dump_json(), now, p['id'], pos))
            if pos == len(manifest)-1:
                c.execute('UPDATE participants SET completed=? WHERE id=?', (now, p['id']))
            result = study_state(c, get_participant(c, p['id']))
        return result

    @app.get('/api/images/{alias}/{patch_alias}/{kind}')
    def image(alias: str, patch_alias: str, kind: Literal['patch', 'frame'], p=Depends(identity)):
        pos = position(alias)
        patch = assignment_patch(p, pos, patch_alias)
        if kind == 'frame' and not cfg.frame_context:
            raise HTTPException(403, 'Full-frame context is disabled for this study.')
        with db() as c:
            require_current(c, p['id'], pos)
            row = c.execute('SELECT * FROM samples WHERE participant=? AND position=?', (p['id'], pos)).fetchone()
            if not row:
                raise HTTPException(409, 'Start this sample first.')
            if kind == 'frame':
                viewed = sorted(set(json.loads(row['frame_views'])) | {patch_alias})
                c.execute('UPDATE samples SET frame_views=? WHERE participant=? AND position=?',
                          (json.dumps(viewed), p['id'], pos))
        path = (cfg.data_root / patch[kind]['path']).resolve()
        if not path.is_relative_to(cfg.data_root):
            raise HTTPException(404, 'Image not found.')
        return FileResponse(path, media_type='image/png' if kind == 'patch' else 'image/jpeg')

    @app.get('/api/export/{filename}')
    def download(filename: Literal['annotations.csv', 'metadata.json'], p=Depends(identity)):
        with db() as c:
            export(c, get_participant(c, p['id']))
        return FileResponse(cfg.annotation_root / ('participant_' + p['id']) / filename,
                            media_type='text/csv' if filename.endswith('.csv') else 'application/json', filename=filename)

    app.mount('/', StaticFiles(directory=REPO / 'docs', html=True), name='frontend')
    return app
