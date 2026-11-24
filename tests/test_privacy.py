"""Synthetic release fixtures, never actual host strings or credentials."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from robofate_pipeline import deployment, privacy


class PrivacyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "module.py").write_text("VALUE = 1\n")

    def audit(self):
        return privacy.audit_tree(self.root, verify_index=False)

    def test_clean_text_is_allowed(self):
        self.assertTrue(self.audit()["passed"])

    def test_every_file_is_indexed_and_hash_checked(self):
        privacy.write_index(self.root)
        self.assertTrue(privacy.audit_tree(self.root)["passed"])
        (self.root / "module.py").write_text("VALUE = 2\n")
        self.assertFalse(privacy.audit_tree(self.root)["passed"])

    def test_added_file_is_not_silently_ignored(self):
        privacy.write_index(self.root)
        (self.root / ".hidden").write_text("extra")
        self.assertFalse(privacy.audit_tree(self.root)["passed"])

    def test_absent_index_is_failure(self):
        self.assertFalse(privacy.audit_tree(self.root)["passed"])

    def test_removed_file_is_failure(self):
        privacy.write_index(self.root)
        (self.root / "module.py").unlink()
        self.assertFalse(privacy.audit_tree(self.root)["passed"])

    def test_git_history_is_not_part_of_release(self):
        (self.root / ".git").mkdir()
        (self.root / ".git/HEAD").write_text("synthetic")
        self.assertFalse(self.audit()["passed"])

    def test_archive_cannot_hide_historical_content(self):
        with tarfile.open(self.root / "source.tar.gz", "w:gz") as archive:
            member = tarfile.TarInfo("nested.txt")
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
        self.assertFalse(self.audit()["passed"])

    def test_symlinks_are_rejected(self):
        (self.root / "alias").symlink_to("module.py")
        self.assertFalse(self.audit()["passed"])

    def test_unknown_binary_is_rejected(self):
        (self.root / "cache.dat").write_bytes(b"\xff\xfe\x00")
        self.assertFalse(self.audit()["passed"])

    def test_runtime_record_is_rejected(self):
        (self.root / "result.json").write_text("{}")
        self.assertFalse(self.audit()["passed"])

    def test_credential_pattern_is_rejected_without_echo(self):
        token = "ghp_" + "Z" * 36
        (self.root / "settings.py").write_text("token = " + repr(token))
        result = self.audit()
        self.assertFalse(result["passed"])
        self.assertNotIn(token, json.dumps(result))

    def test_personal_home_is_rejected_without_echo(self):
        value = "/" + "Users" + "/" + "synthetic-person" + "/project"
        (self.root / "settings.py").write_text(repr(value))
        result = self.audit()
        self.assertFalse(result["passed"])
        self.assertNotIn(value, json.dumps(result))

    def test_private_literal_scans_names_and_binary_bytes(self):
        (self.root / "SYNTHETIC_PRIVATE.py").write_text("pass")
        result = privacy.audit_tree(self.root, denylist=["SYNTHETIC_PRIVATE"], verify_index=False)
        self.assertFalse(result["passed"])
        self.assertNotIn("SYNTHETIC_PRIVATE", json.dumps(result))

    def test_empty_release_is_failure(self):
        (self.root / "module.py").unlink()
        self.assertFalse(self.audit()["passed"])

    def test_host_checks_do_not_grant_launch(self):
        result = deployment.inspect({}, hostname="synthetic-host")
        self.assertFalse(result["static_checks_passed"])
        self.assertFalse(result["collection_ready"])
        self.assertFalse(result["live_collection_validated"])

    def test_synthetic_metadata_cannot_authorize_collection(self):
        result = deployment.inspect({"expected_hostname": "synthetic-host",
                                     "execution_owner": "synthetic",
                                     "data_kind": "synthetic"}, hostname="synthetic-host")
        self.assertIn("Synthetic metadata cannot authorize real collection", result["errors"])
