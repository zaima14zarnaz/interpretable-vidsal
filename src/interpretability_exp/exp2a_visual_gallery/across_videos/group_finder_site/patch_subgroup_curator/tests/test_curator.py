import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location("curator", Path(__file__).resolve().parents[1] / "patch_subgroup_curator.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def patch(index, ids):
    return {"group_patch_index": index, "video_id": f"v{index}",
            "frame_filename": f"{index:05d}.png", "grid_row": 2, "grid_column": 3,
            "patch_directory": f"group_03/patch_{index:02d}", "extra_metadata": {"retained": True},
            "top_k_cosine_matches": [{"prototype_index": pid, "cosine_similarity": score,
                "actual_valid_active": i == 0, "forward_activation": .12 + i}
                for i, (pid, score) in enumerate(ids)]}


class TestCurator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "gallery"
        self.root.mkdir()
        self.patches = [patch(0, [(9, .4), (3, .9), (1, .7)]),
                        patch(2, [(1, .8), (9, .7), (3, .8)]),
                        patch(7, [(3, .95), (4, .8), (9, .5)])]
        self.group = self.root / "group_03"
        self.group.mkdir()
        self.path = self.group / "patch_metadata.json"
        self.path.write_text(json.dumps(self.patches))
        for p in self.patches:
            folder = self.root / p["patch_directory"]
            folder.mkdir()
            (folder / "patch.png").write_bytes(b"sample-patch")
            (folder / "target_frame_boxed.png").write_bytes(b"sample-frame")
        self.output = Path(self.tmp.name) / "saved"
        self.app = app.Curator(self.root, self.output)

    def tearDown(self):
        self.app.close()
        self.tmp.cleanup()

    def body(self, ids=None, k=2):
        return {"source_group": "group_03", "patch_ids": [0, 2] if ids is None else ids,
                "k": k, "name": "Diagonal edges", "description": "A test label",
                "metadata_sha256": self.app.load("group_03")[0]}

    def test_overlap_reference_order_and_stable_ties(self):
        report = app.overlap_report(self.patches[:2], 2)
        self.assertEqual(report["shared_by_all_prototype_ids"], [3, 1])
        self.assertEqual([m["prototype_index"] for m in report["per_patch"][1]["top_k_cosine_matches"]], [1, 3])
        self.assertEqual(report["pairwise"][0]["jaccard"], 1)
        report = app.overlap_report(self.patches, 3)
        self.assertEqual(report["shared_by_all_prototype_ids"], [3, 9])
        support = {r["prototype_index"]: r for r in report["prototype_support"]}
        self.assertEqual(support[3]["patch_ids"], [0, 2, 7])

    def test_empty_single_missing_ranks_and_no_overlap(self):
        self.assertEqual(app.overlap_report([], 20)["shared_by_all_prototype_ids"], [])
        self.assertEqual(app.overlap_report(self.patches[:1], 20)["shared_by_all_prototype_ids"], [3, 1, 9])
        self.assertEqual(app.overlap_report(self.patches[:1], 20)["per_patch"][0]["available_match_count"], 3)
        self.assertEqual(app.overlap_report([self.patches[0], patch(88, [(88, .9)])], 20)["shared_by_all_prototype_ids"], [])

    def test_save_preserves_metadata_and_restart_locks(self):
        original = self.path.read_bytes()
        result = self.app.form(self.body())
        saved = json.loads(Path(result["saved_path"]).read_text())
        self.assertEqual(saved["patches"][0]["metadata"], self.patches[0])
        self.assertEqual(saved["shared_by_all_prototype_ids"], [3, 1])
        self.assertEqual(saved["patches"][0]["selected_top_k_cosine_matches"][0]["prototype_index"], 3)
        self.assertEqual(self.path.read_bytes(), original)
        restarted = app.Curator(self.root, self.output)
        self.assertEqual(restarted.group("group_03")["locked_patch_ids"], {"0": "subgroup_0001", "2": "subgroup_0001"})
        with self.assertRaises(app.Problem) as cm:
            restarted.form(self.body([0, 7]))
        self.assertEqual(cm.exception.status, 409)
        second = self.app.form(self.body([7], 20))
        saved2 = json.loads(Path(second["saved_path"]).read_text())
        self.assertEqual(saved2["subgroup_id"], "subgroup_0002")
        self.assertEqual(saved2["patches"][0]["effective_top_k"], 3)

    def test_concurrent_save_has_one_winner(self):
        results = []
        barrier = threading.Barrier(2)
        body = self.body()
        def save():
            barrier.wait()
            try:
                self.app.form(body)
                results.append(201)
            except app.Problem as exc:
                results.append(exc.status)
        threads = [threading.Thread(target=save) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(results), [201, 409])
        self.assertEqual(len(self.app.saved("group_03")), 1)

    def test_changed_export_allows_new_subgroups(self):
        body = self.body()
        self.path.write_text(json.dumps(self.patches, indent=2))
        with self.assertRaises(app.Problem) as cm:
            self.app.form(body)
        self.assertEqual(cm.exception.status, 409)
        self.app.form(self.body())
        self.path.write_text(json.dumps(self.patches, indent=4))
        self.assertEqual(self.app.group("group_03")["stale_saved_subgroups"], ["subgroup_0001"])
        second = self.app.form(self.body([7]))
        self.assertEqual(second["subgroup"]["subgroup_id"], "subgroup_0002")

    def test_validation_and_process_lock(self):
        for ids, k in [([], 2), ([0, 0], 2), ([999], 2), ([True], 2), ([0], 0)]:
            with self.assertRaises(app.Problem):
                self.app.form(self.body(ids, k))
        with self.assertRaises(app.Problem):
            app.Curator(self.root, self.root / "subgroups")
        self.app.acquire_process_lock()
        other = app.Curator(self.root, self.output)
        with self.assertRaises(app.Problem):
            other.acquire_process_lock()

    def test_numeric_group_order_and_corrupt_saved_file(self):
        for name in ["group_10", "group_2"]:
            folder = self.root / name
            folder.mkdir()
            (folder / "patch_metadata.json").write_text("[]")
        self.assertEqual([g["name"] for g in self.app.groups()["groups"]], ["group_2", "group_03", "group_10"])
        self.app.form(self.body())
        (self.output / "group_03" / "subgroup_0001.json").write_text("broken")
        with self.assertRaises(app.Problem):
            self.app.group("group_03")

    def test_http_routes_save_download_and_traversal(self):
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        server.curator = self.app
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base) as res:
                html = res.read()
                self.assertIn(b'grid-template-columns:repeat(8', html)
                self.assertIn(b'id="activation-freq-details"', html)
                self.assertIn(b"Activation frequency", html)
            with urlopen(base + "/api/groups") as res:
                config = json.load(res)
            self.assertIn("activation_frequencies", config)
            with urlopen(base + "/api/group?name=group_03") as res:
                group = json.load(res)
            self.assertTrue(group["patches"][0]["media"]["frame"].endswith("target_frame_boxed.png"))
            with urlopen(base + group["patches"][0]["media"]["patch"]) as res:
                self.assertEqual(res.read(), b"sample-patch")
            data = json.dumps(self.body()).encode()
            with self.assertRaises(HTTPError) as cm:
                urlopen(Request(base + "/api/subgroups", data=data, headers={"Content-Type": "application/json"}))
            self.assertEqual(cm.exception.code, 403)
            request = Request(base + "/api/subgroups", data=data, headers={"Content-Type": "application/json", "X-Curator-Token": config["csrf_token"]})
            with urlopen(request) as res:
                self.assertEqual(res.status, 201)
                result = json.load(res)
            with urlopen(base + result["download_url"]) as res:
                self.assertIn("attachment", res.headers["Content-Disposition"])
                self.assertEqual(json.load(res)["patch_ids"], [0, 2])
            with self.assertRaises(HTTPError):
                urlopen(base + "/media/group_03/../../saved/group_03/subgroup_0001.json")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_activation_frequency_csv_loaded_and_served(self):
        csv_path = Path(self.tmp.name) / "prototype_activation_freq_test.csv"
        csv_path.write_text(
            "prototype_index,train_count,train_frequency,val_count,val_frequency,"
            "combined_count,combined_frequency\n"
            "3,10,0.1,5,0.05,15,0.075\n"
            "1,20,0.2,10,0.1,30,0.15\n"
        )
        curator = app.Curator(self.root, self.output / "freq_out", activation_freq_csv=csv_path)
        try:
            self.assertEqual(curator.activation_frequencies["source_csv"], str(csv_path.resolve()))
            proto = curator.activation_frequencies["prototypes"]["3"]
            self.assertEqual(proto["combined_count"], 15)
            self.assertAlmostEqual(proto["combined_frequency"], 0.075)
            payload = curator.groups()
            self.assertEqual(payload["activation_frequencies"]["prototypes"]["1"]["train_count"], 20)
        finally:
            curator.close()
        with self.assertRaises(app.Problem):
            app.Curator(self.root, self.output / "missing_csv",
                        activation_freq_csv=Path(self.tmp.name) / "does_not_exist.csv")


if __name__ == "__main__":
    unittest.main()
