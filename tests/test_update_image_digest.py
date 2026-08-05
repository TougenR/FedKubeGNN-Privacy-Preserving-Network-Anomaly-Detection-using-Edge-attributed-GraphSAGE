from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.update_image_digest import update


class UpdateImageDigestTest(unittest.TestCase):
    def test_updates_only_first_digest_and_release(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "values.yaml"
            path.write_text(
                "image:\n  digest: sha256:" + "0" * 64 + "\nreleaseId: old-release\n",
                encoding="utf-8",
            )
            digest = "sha256:" + "a" * 64
            update(path, digest, "abc1234")
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                f"image:\n  digest: {digest}\nreleaseId: abc1234\n",
            )


if __name__ == "__main__":
    unittest.main()
