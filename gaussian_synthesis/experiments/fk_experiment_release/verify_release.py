"""Verify a published FK archive without changing frozen experiment files.

Usage: python verify_release.py --output /tmp/fk-release-audit-new
The output directory must not exist and must be outside the three frozen
experiment packages. No sampling or optimization is performed. Optional
--integrity-only skips the deterministic scientific audit, never upgrades a
historical development run to a verified scientific result.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PACKAGES = ("fk_two_cluster", "fk_two_cluster_followup", "fk_canonical_structure")
EXPERIMENTS = Path("gaussian_synthesis/experiments")
CANONICAL_RUN = EXPERIMENTS/"fk_canonical_structure/results/canonical_20261024_01"


def _write_json(path, document):
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _under(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _valid_digest(value, length):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def check_manifest(manifest_path, repository):
    """Read every listed byte and every NPZ member; return explicit failures."""
    repository = Path(repository).resolve()
    manifest_path = Path(manifest_path)
    failures, rows = [], []
    total_bytes = npz_files = npz_members = 0
    try:
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
            raise ValueError("expected schema_version=1 and a files array")
        if not manifest["files"]:
            raise ValueError("file inventory must not be empty")
    except Exception as exc:
        return {"status": "failed", "files_checked": 0, "failures": [
            {"check": "manifest_schema", "error": str(exc)}], "files": []}
    seen = set()
    for record in manifest["files"]:
        label = record.get("path", "<missing>") if isinstance(record, dict) else "<invalid>"
        row = {"path": label, "status": "failed"}
        try:
            if not isinstance(record, dict) or not isinstance(label, str):
                raise ValueError("file entries require a string path")
            relative = Path(label)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("file path must be repository-relative without '..'")
            if label in seen:
                raise ValueError("duplicate file path")
            seen.add(label)
            expected_size = record.get("size_bytes")
            if type(expected_size) is not int or expected_size < 0:
                raise ValueError("size_bytes must be a nonnegative integer")
            if not _valid_digest(record.get("sha256"), 64) or not _valid_digest(record.get("git_blob_sha"), 40):
                raise ValueError("expected lowercase SHA256 and Git blob SHA-1 digests")
            path = (repository/relative).resolve()
            if not _under(path, repository):
                raise ValueError("file or symlink escapes the repository")
            if not path.is_file():
                raise FileNotFoundError("listed release file is missing")
            size = path.stat().st_size
            sha256 = hashlib.sha256()
            git_blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
            with path.open("rb") as stream:
                while block := stream.read(1024*1024):
                    sha256.update(block)
                    git_blob.update(block)
            row.update(size_bytes=size, sha256=sha256.hexdigest(), git_blob_sha=git_blob.hexdigest())
            discrepancies = [field for field in ("size_bytes", "sha256", "git_blob_sha") if row[field] != record[field]]
            if discrepancies:
                raise ValueError("manifest mismatch: "+", ".join(discrepancies))
            total_bytes += size
            if path.suffix.lower() == ".npz":
                import numpy as np
                with zipfile.ZipFile(path) as archive:
                    bad_member = archive.testzip()
                    if bad_member is not None:
                        raise ValueError("ZIP CRC failure: "+bad_member)
                members = []
                with np.load(path, allow_pickle=False) as archive:
                    for name in archive.files:
                        array = archive[name]
                        members.append({"name": name, "shape": list(array.shape), "dtype": str(array.dtype)})
                        del array
                row["npz_members"] = members
                row["npz_crc_checked"] = True
                npz_files += 1
                npz_members += len(members)
            row["status"] = "passed"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures.append({"check": "file_integrity", "path": label, "error": row["error"]})
        rows.append(row)
    return {"status": "passed" if not failures else "failed", "files_checked": len(rows),
            "verified_bytes": total_bytes, "npz_files_checked": npz_files,
            "npz_members_read": npz_members, "failures": failures, "files": rows,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "scope": "Listed files only; extra files outside the inventory are not certified by this check."}


def historical_status(repository):
    """Report stored history without rerunning, changing, or endorsing it."""
    rows = []
    for package in PACKAGES:
        result_root = Path(repository)/EXPERIMENTS/package/"results"
        if not result_root.is_dir():
            continue
        for run in sorted(result_root.iterdir()):
            if not run.is_dir():
                continue
            row = {"run": str(run.relative_to(repository)), "historical_record_only": True,
                   "scientific_success_inferred": False}
            for filename, field in (("summary.json", "stored_run_status"), ("verification.json", "stored_verification_status")):
                path = run/filename
                if path.is_file():
                    try:
                        data = _read_json(path)
                        value = data.get("status")
                        if filename == "verification.json" and value is None and "audit_passed" in data:
                            value = "passed" if data["audit_passed"] else "failed"
                        row[field] = value
                        if filename == "verification.json":
                            row["stored_verification_failures"] = data.get("failures", [])
                    except Exception as exc:
                        row[field] = "unreadable"
                        row["error"] = str(exc)
            manifest = run/"manifest.json"
            if package == "fk_two_cluster_followup" and manifest.is_file():
                try:
                    source = str(_read_json(manifest).get("source_run", ""))
                    row["stored_source_run"] = source
                    row["source_run_is_absolute"] = Path(source).is_absolute()
                except Exception as exc:
                    row["source_manifest_error"] = str(exc)
                row["rerun_policy"] = "Not rerun: the legacy verifier resolves its historical absolute source_run; development source hashes can differ from the released code."
            if "smoke" in run.name:
                row["interpretation"] = "Development snapshot; source hash differences after later edits are retained, not repaired or interpreted as a formal successful run."
            rows.append(row)
    return rows


def canonical_scientific_audit(repository, output):
    """Call the frozen audit with its sole JSON writer redirected externally."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    source = repository/EXPERIMENTS/"fk_canonical_structure/scripts/verify_results.py"
    run = repository/CANONICAL_RUN
    target = output/"canonical_verification.json"
    result = {"status": "failed", "reruns_sampling": False, "reruns_optimization": False,
              "input_run": str(CANONICAL_RUN), "output_file": target.name,
              "strict_interval_certified": False}
    try:
        spec = importlib.util.spec_from_file_location("_fk_release_frozen_canonical_verifier", source)
        if spec is None or spec.loader is None:
            raise ImportError("cannot load frozen canonical verifier")
        module = importlib.util.module_from_spec(spec)
        original_verification = run/"verification.json"
        original_hash = (hashlib.sha256(original_verification.read_bytes()).hexdigest()
                         if original_verification.is_file() else None)
        with (output/"canonical_verification.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                spec.loader.exec_module(module)

                def redirected_save(path, document):
                    if Path(path).resolve() != original_verification.resolve():
                        raise RuntimeError("frozen verifier attempted an unexpected output path")
                    _write_json(target, document)

                module.save_json = redirected_save
                verification = module.verify(run)
                print(json.dumps(verification, indent=2))
        after_hash = (hashlib.sha256(original_verification.read_bytes()).hexdigest()
                      if original_verification.is_file() else None)
        if after_hash != original_hash:
            raise RuntimeError("published verification file changed unexpectedly")
        result.update(status=verification["status"], checks=verification.get("checks"),
                      failures=verification.get("failures", []), original_verification_unchanged=True,
                      limits=verification.get("limits", []))
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        with (output/"canonical_verification.log").open("a", encoding="utf-8") as log:
            log.write(result["traceback"])
    return result


def verify_release(manifest_path, repository, output, *, integrity_only=False):
    """Audit entry point, also usable with small temporary test fixtures."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    if any(_under(output, repository/EXPERIMENTS/package) for package in PACKAGES):
        raise ValueError("output directory must be outside the three frozen experiment packages")
    output.mkdir(parents=True)
    started = time.monotonic()
    integrity = check_manifest(manifest_path, repository)
    _write_json(output/"integrity.json", integrity)
    history = historical_status(repository)
    _write_json(output/"historical_status.json", history)
    if integrity["status"] != "passed":
        scientific = {"status": "skipped_integrity_failed", "reruns_sampling": False, "reruns_optimization": False}
    elif integrity_only:
        scientific = {"status": "not_requested", "reruns_sampling": False, "reruns_optimization": False}
    else:
        scientific = canonical_scientific_audit(repository, output)
    success = integrity["status"] == "passed" and (integrity_only or scientific["status"] == "passed")
    result = {
        "status": "passed_integrity_only" if success and integrity_only else "passed" if success else "failed",
        "repository": str(repository), "output": str(output),
        "integrity_status": integrity["status"], "files_checked": integrity["files_checked"],
        "npz_files_checked": integrity.get("npz_files_checked", 0),
        "npz_members_read": integrity.get("npz_members_read", 0),
        "integrity_failure_count": len(integrity["failures"]),
        "canonical_scientific_audit": scientific, "historical_runs_reported": len(history),
        "elapsed_seconds": time.monotonic()-started, "numerical_processes": 1, "numerical_threads": 1,
        "published_files_modified": False, "strict_interval_certified": False,
        "interpretation": "Archive integrity and deterministic saved-array checks are distinct from scientific acceptance, statistical confidence, and global Gaussian optimality. Historical failed or limited outcomes remain unchanged.",
    }
    _write_json(output/"release_verification.json", result)
    (output/"verification.log").write_text(
        f"status={result['status']}\nfiles_checked={result['files_checked']}\n"
        f"npz_files_checked={result['npz_files_checked']}\nnpz_members_read={result['npz_members_read']}\n"
        f"integrity_failure_count={result['integrity_failure_count']}\n"
        f"scientific_status={scientific['status']}\n"
        "No sampling or optimization. Published experiment files were not intentionally written.\n",
        encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new audit output directory outside frozen experiments")
    parser.add_argument("--integrity-only", action="store_true", help="verify archive bytes and NPZ members without the scientific audit")
    args = parser.parse_args()
    try:
        result = verify_release(HERE/"release_manifest.json", REPOSITORY, args.output,
                                integrity_only=args.integrity_only)
    except Exception as exc:
        print(f"Verification could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("passed", "passed_integrity_only") else 1


if __name__ == "__main__":
    raise SystemExit(main())
