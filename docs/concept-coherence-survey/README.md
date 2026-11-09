# Visual Pattern Study

A GitHub Pages frontend plus a Python backend for **Experiment 1: unlabeled visual coherence**.
Each sample is a **concept group**, not an individual patch. Every invited participant evaluates every discovered group. Titles, author descriptions, activation scores, real group IDs, and real patch IDs stay hidden during annotation.

## What participants do

1. Enter their email and click **Enter survey** (no verification code).
2. Read neutral study instructions.
3. Inspect the patches in one group, optionally opening the boxed full frames.
4. Describe the shared pattern, or enter **No clear shared pattern**.
5. Rate clarity: **1 No discernible pattern · 2 Weak · 3 Moderately clear · 4 Clear · 5 Very clear**.
6. Identify outlier patches, choose **None**, or choose **Cannot determine**.
7. Submit and proceed. The last submission automatically writes the final CSV and metadata JSON on the dataset server.

Group order and patch order are randomized independently for each participant and preserved on resume. All participants see the same image sets. Submitted answers are locked. Drafts save after a short typing pause; a network-interrupted draft also stays in the current browser tab. Re-signing in with the same email restores server-saved progress. Closing a tab can lose its last unsynced edit or timing interval.

## Why there are two pieces

GitHub Pages serves static HTML/CSS/JavaScript. It cannot read or write `/data/...` on your research machine or run the Python backend. Run the backend **on the machine that can access the dataset**, and expose it through HTTPS. GitHub Pages then calls that backend. Images are fetched only after authentication; you do not need to upload DHF1K images or annotations to GitHub.

For initial testing you can open the frontend directly from the backend. This uses the same UI and needs no GitHub configuration.

## Dataset layout

Defaults are already set to your paths:

```text
/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v5/
    group_0/
        patch_0/
            patch.png
            target_frame_boxed.png
        patch_1/
            patch.png
            target_frame_boxed.png
    group_1/
        ...
    annotations/
```

Group and patch suffixes need not be consecutive. All `group_* / patch_*` directories are discovered automatically. **Both image files must exist for every patch**; a missing image stops startup rather than silently changing the study population. Set `GROUP_IDS=0,1,2,...` to restrict which group suffixes participate. This is useful if the directory also contains unfinished groups. No title/description files are read.

At startup the backend freezes a manifest containing group IDs, patch IDs, image hashes, protocol version, and frame-context setting. Later restarts reject changed assets/configuration for the same annotation directory. Keep the source images unchanged while collecting annotations. For a new image set or protocol, use a new annotation directory. The manifest and database are stored in `annotations/.study_state/`; retain that entire folder to preserve identities and resume state.

## 1. Install on the dataset server

Python **3.10 or newer** is required. Run from this repository's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```dotenv
DATA_ROOT=/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v5
ANNOTATION_ROOT=/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v5/annotations
GROUP_IDS=
ALLOW_FRAME_CONTEXT=true
ALLOWED_ORIGINS=https://YOUR-GITHUB-USERNAME.github.io
AUTH_MODE=email_only
```

**Default `AUTH_MODE=email_only`:** participants type any email, click **Enter survey**, and continue immediately. No verification codes, invitation allowlist, domain checks, or SMTP. The email is stored in participant metadata with `email_verified=false`. Returning with the same normalized email resumes that participant. Different emails create separate records under `ANNOTATION_ROOT/participant_<id>/`.

Legacy modes: `preview` (anonymous test data under `ANNOTATION_ROOT/preview/`), `console` / `smtp` (verification codes; require `ALLOWED_EMAILS`, and `smtp` also needs SMTP settings). SMTP credentials belong **only in the server's `.env`**, never in `docs/config.js`.

Sessions last 24 hours; participants can enter the same email again to resume. The server process needs read permission for source images and write permission for the annotation directory. For a small study use one Uvicorn worker, as below. SQLite should be on a filesystem that reliably supports its locking semantics; avoid unreliable multi-host network mounts. Back up the private annotation directory.

## 2. Test the real dataset locally

### Email-only mode (default)

```bash
cd /home/z/zaimazarnaz/research1/ExplainableSaliency/docs/concept-coherence-survey
AUTH_MODE=email_only \
DATA_ROOT=/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v5 \
ANNOTATION_ROOT=/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v5/annotations \
ALLOWED_ORIGINS=https://zaima14zarnaz.github.io \
python -m uvicorn backend.app:create_app --factory --host 127.0.0.2 --port 8000
```

Open the backend URL or the GitHub Pages frontend and enter any email. No code is sent.

### Preview mode (anonymous test annotations)

```bash
AUTH_MODE=preview \
DATA_ROOT=/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v5 \
ANNOTATION_ROOT=/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v5/annotations \
ALLOWED_ORIGINS=https://zaima14zarnaz.github.io \
python -m uvicorn backend.app:create_app --factory --host 127.0.0.2 --port 8000
```

Click **Start preview**. Use **Start a new preview** for a fresh anonymous participant. Preview exports stay under `ANNOTATION_ROOT/preview/` and are marked `is_test_data`.

### Console / SMTP code flows (legacy)

`AUTH_MODE=console` prints codes in the terminal; `AUTH_MODE=smtp` emails them. Both require `ALLOWED_EMAILS`.

```bash
python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000
```

Keep `docs/config.js` `apiBase: ""` for same-origin testing against that process.

## 3. Give the backend an HTTPS address

The backend must be reachable from participants' browsers. A typical address is `https://survey-api.your-university.edu`. Keep Uvicorn bound to `127.0.0.1:8000` and put an HTTPS reverse proxy in front of it. `deploy/Caddyfile.example` includes a minimal Caddy configuration. Your university's existing HTTPS proxy also works. You need a domain routed to that server and allowed ingress; a GitHub Pages URL alone does not provide this connectivity.

An HTTPS GitHub Pages frontend needs an **HTTPS API**. Do not point it to an HTTP API or the server's private `/data` path. Keep the backend process running for the duration of the study, using your institution's service manager or a persistent terminal session.

## 4. Publish the frontend on GitHub Pages

1. Create a repository and upload this source tree. Do not upload `.env`, images, databases, or annotation output.
2. Edit `docs/config.js`:

   ```javascript
   window.SURVEY_CONFIG = Object.freeze({
     apiBase: "https://survey-api.your-university.edu"
   });
   ```

3. In the server `.env`, set `ALLOWED_ORIGINS=https://YOUR-GITHUB-USERNAME.github.io`. This is the origin **without** the repository path. For a custom frontend domain, use its exact origin instead. Restart the backend after configuration changes.
4. In GitHub, open **Settings → Pages → Deploy from a branch**. Select your main branch and the **`/docs`** folder, then save.
5. Open `https://YOUR-GITHUB-USERNAME.github.io/YOUR-REPOSITORY/` and test one invited email end to end.

The repository includes a GitHub Actions test workflow; it does not deploy your backend. GitHub sign-in is not required for annotators.

## Saved files

After the final group, files are written automatically to:

```text
/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v5/annotations/
    participant_<random-participant-id>/
        annotations.csv
        metadata.json
```

The participant ID is a random stable identifier mapped to their email in the private database. Reusing the same email resumes the same record. The email is in `metadata.json`, not repeated in every CSV row. The participant can download only their own final files after completion.

### CSV: one row per concept group

| Column | Meaning |
|---|---|
| `participant_id` | Stable random participant ID |
| `study_sha256` | Hash of the frozen study manifest |
| `sample_order`, `sample_alias` | Participant-specific presentation position and neutral alias |
| `group_id` | Original group directory suffix for analysis |
| `patch_order_json` | Display patch ID → original patch directory suffix |
| `description` | Participant's independent description |
| `coherence` | Ordinal clarity rating, 1–5 |
| `outlier_mode` | `selected`, `none`, or `cannot_determine` |
| `outlier_patch_ids_json` | Original IDs of selected outlier patches |
| `outlier_display_ids_json` | IDs shown in the survey for those outliers |
| `frame_viewed_display_ids_json` | Unique patches whose full-frame image was requested |
| `started_at_utc`, `submitted_at_utc` | Server timestamps |
| `active_seconds` | Recorded foreground time on that group |
| `elapsed_seconds` | Wall time from first opening that group to submission, including breaks |

List/dictionary fields are JSON inside normal CSV cells. Free text beginning with spreadsheet formula characters is prefixed with an apostrophe in the CSV to prevent formula execution; the original response remains in the private SQLite database.

### Metadata JSON

Includes email, participant ID, registration/start/completion timestamps, total groups/patches, protocol/hash, randomized assignment, frame-context policy, and:

| Field | Definition |
|---|---|
| `total_annotation_time_seconds` | Sum of recorded active time across all groups |
| `average_time_per_sample_seconds` | Active time divided by number of concept groups |
| `total_elapsed_time_seconds` | Wall time from first group opening to final submission, including breaks |
| `average_elapsed_time_per_sample_seconds` | Mean of individual groups' opening-to-submission wall times |

Active time counts a visible, focused survey tab, including reading and thinking. It pauses on blur or when the tab is hidden. The browser sends timing intervals every five seconds, and the server caps each interval by wall time and 15 seconds. A browser crash, a prolonged connection failure, or a device suspend can lose time; this is an estimate of foreground time, not a measure of attention. Both active and elapsed timings are provided to avoid treating breaks as annotation effort.

Drafts and submissions are persisted in SQLite throughout the study. Final files use atomic replacement, with metadata written last and carrying the CSV checksum. An interrupted export can be retried or regenerated from SQLite by signing in and loading the completed study. The completion screen appears only after both final files have been written successfully.

## Full-frame policy

Full frames are available because you supplied their paths. With `ALLOW_FRAME_CONTEXT=true`, participants choose whether to open them and each requested frame is logged. This evaluates coherence **with optional context**. Set it to `false` before starting the final study if your claim specifically concerns recognition from patches alone. The policy is frozen with the study manifest and cannot change midway through the same study.

## Tests and a synthetic demo

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
node --check docs/app.js
```

The integration tests cover authentication, participant isolation, draft recovery, stable assignments, answer validation, stale-tab protection, timing, frame-view logging, completion exports, concurrent/repeated submission, export recovery after a simulated write failure, and unsafe assets/changed manifests. They use a temporary dataset and never touch your real data.

For a clearly synthetic UI demo, install Pillow and generate a new test directory:

```bash
python -m pip install Pillow
python scripts/make_demo.py /tmp/coherence-demo
```

Then start in the same shell with explicit test settings:

```bash
DATA_ROOT=/tmp/coherence-demo ANNOTATION_ROOT=/tmp/coherence-demo/annotations ALLOWED_EMAILS=demo@example.edu AUTH_MODE=console GROUP_IDS= python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000
```

Open the backend URL with `apiBase: ""`, sign in as `demo@example.edu`, and copy the one-time code from the terminal. The demo contains three artificial groups and is not research data.

## Study interpretation

This app tests independent recognizability and perceived coherence of the displayed groups. If the patches were manually curated, report that selection procedure. Coherence ratings alone do not establish that all activations of a prototype share the pattern or that the pattern causally explains saliency. Use all three raters independently and keep submitted responses unchanged when reporting agreement.
