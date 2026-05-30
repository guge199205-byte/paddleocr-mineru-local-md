import importlib
import os
import tempfile
import unittest

from fastapi.testclient import TestClient


class ServerTaskApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        os.environ["PANDOCR_TASK_DATA_DIR"] = cls.temp_dir.name
        os.environ["PANDOCR_MAX_UPLOAD_MB"] = "1"
        cls.server = importlib.import_module("server")
        cls.client = TestClient(cls.server.app)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_task_list_returns_summaries_and_detail_endpoint_returns_full_task(self):
        task = {
            "id": "task_123",
            "name": "sample.pdf",
            "sourceKind": "pdf",
            "size": 1200,
            "createdAt": 100,
            "updatedAt": 200,
            "status": "processing",
            "pageCount": 3,
            "sourceDataUrl": "data:application/pdf;base64,JVBERi0=",
            "batches": [
                {"id": "b1", "status": "completed", "pageCount": 1},
                {"id": "b2", "status": "pending", "pageCount": 2},
            ],
            "markdown": "# Result",
            "images": {"ocr_images/a.jpg": "abc"},
            "ocrResults": [{"markdown": {"text": "# Result"}}],
        }

        put_response = self.client.put("/api/tasks/task_123", json=task)
        self.assertEqual(put_response.status_code, 200)

        list_response = self.client.get("/api/tasks")
        self.assertEqual(list_response.status_code, 200)
        summary = list_response.json()["tasks"][0]
        self.assertEqual(summary["id"], "task_123")
        self.assertEqual(summary["completedPages"], 1)
        self.assertTrue(summary["hasMarkdown"])
        self.assertNotIn("sourceDataUrl", summary)
        self.assertNotIn("batches", summary)
        self.assertNotIn("ocrResults", summary)

        detail_response = self.client.get("/api/tasks/task_123")
        self.assertEqual(detail_response.status_code, 200)
        detail = detail_response.json()
        self.assertEqual(detail["sourceDataUrl"], task["sourceDataUrl"])
        self.assertEqual(detail["batches"], task["batches"])
        self.assertTrue(detail["detailLoaded"])

    def test_invalid_task_id_is_rejected(self):
        response = self.client.get("/api/tasks/bad!")
        self.assertEqual(response.status_code, 400)

    def test_oversized_request_is_rejected_before_proxying(self):
        large_payload = {"image": "x" * (2 * 1024 * 1024), "fileType": 1}
        response = self.client.post("/api/paddleocr-vl-1.6", json=large_payload)
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
