import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter


class ServerTaskApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        os.environ["PANDOCR_TASK_DATA_DIR"] = cls.temp_dir.name
        os.environ["PANDOCR_MAX_UPLOAD_MB"] = "1"
        os.environ["PANDOCR_MODEL_CONTROL"] = "none"
        os.environ["PANDOCR_API_TOKEN"] = ""
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
            "modelId": "pp-ocrv6",
            "modelName": "PP-OCRv6",
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
        self.assertEqual(summary["modelId"], "pp-ocrv6")
        self.assertEqual(summary["modelName"], "PP-OCRv6")
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

    def test_task_list_sorts_mixed_timestamp_formats(self):
        numeric_dir = Path(self.temp_dir.name) / "task_sort_numeric"
        iso_dir = Path(self.temp_dir.name) / "task_sort_iso"
        numeric_dir.mkdir(parents=True, exist_ok=True)
        iso_dir.mkdir(parents=True, exist_ok=True)
        (numeric_dir / "task.json").write_text(
            json.dumps({"id": "task_sort_numeric", "updatedAt": 4102444800}),
            encoding="utf-8",
        )
        (iso_dir / "task.json").write_text(
            json.dumps({"id": "task_sort_iso", "updatedAt": "1970-01-01T00:01:00Z"}),
            encoding="utf-8",
        )

        response = self.client.get("/api/tasks")
        self.assertEqual(response.status_code, 200)
        ids = [task["id"] for task in response.json()["tasks"]]
        self.assertLess(ids.index("task_sort_numeric"), ids.index("task_sort_iso"))

    def test_model_list_includes_vl_and_ppocrv6(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        model_ids = [model["id"] for model in response.json()["data"]]
        self.assertIn("paddleocr-vl-1.6", model_ids)
        self.assertIn("pp-ocrv6", model_ids)

    def test_model_runtime_reports_both_models(self):
        with patch.object(self.server, "check_http_health", new=AsyncMock(return_value=False)):
            response = self.client.get("/api/model-runtime")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("models", payload)
        self.assertIn("paddleocr-vl-1.6", payload["models"])
        self.assertIn("pp-ocrv6", payload["models"])
        self.assertIn("controlAvailable", payload)
        self.assertIn("ocrActiveCount", payload)
        self.assertIn("maxConcurrentOcr", payload)

    def test_model_runtime_switch_requires_docker_control(self):
        with patch.object(self.server, "model_control_available", return_value=False):
            response = self.client.post("/api/model-runtime/switch", json={"modelId": "pp-ocrv6"})
        self.assertEqual(response.status_code, 503)

    def test_cross_origin_mutation_is_rejected_without_allowlisted_origin(self):
        response = self.client.post(
            "/api/model-runtime/switch",
            json={"modelId": "pp-ocrv6"},
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(response.status_code, 403)

    def test_allowlisted_origin_can_reach_api(self):
        with patch.object(self.server, "model_control_available", return_value=False):
            response = self.client.post(
                "/api/model-runtime/switch",
                json={"modelId": "pp-ocrv6"},
                headers={"Origin": "http://localhost:8000"},
            )
        self.assertEqual(response.status_code, 503)

    def test_invalid_task_id_is_rejected(self):
        response = self.client.get("/api/tasks/bad!")
        self.assertEqual(response.status_code, 400)

    def test_oversized_request_is_rejected_before_proxying(self):
        large_payload = {"image": "x" * (2 * 1024 * 1024), "fileType": 1}
        response = self.client.post("/api/paddleocr-vl-1.6", json=large_payload)
        self.assertEqual(response.status_code, 413)

    def test_ppocr_response_is_normalized_for_existing_frontend(self):
        response = self.server.parse_ppocr_response(
            {
                "result": {
                    "ocrResults": [
                        {
                            "inputImage": "base64-page-image",
                            "prunedResult": {
                                "page_index": 0,
                                "rec_texts": ["Hello", "World"],
                                "rec_scores": [0.98, 0.95],
                                "rec_boxes": [[1, 2, 30, 10], [1, 14, 40, 22]],
                            }
                        }
                    ]
                }
            }
        )

        self.assertEqual(response["markdown"], "Hello\nWorld")
        self.assertEqual(len(response["layoutParsingResults"]), 1)
        page = response["layoutParsingResults"][0]
        self.assertEqual(page["parser"], "pp-ocrv6")
        self.assertEqual(page["pageImage"], "base64-page-image")
        self.assertEqual(page["ocrLines"][0]["text"], "Hello")
        self.assertEqual(page["ocrLines"][0]["box"], [1, 2, 30, 10])

    def test_task_source_is_stored_outside_task_json_and_page_ranges_can_be_read(self):
        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(width=72, height=72)
        pdf_buffer = io.BytesIO()
        writer.write(pdf_buffer)
        pdf_bytes = pdf_buffer.getvalue()

        upload_response = self.client.post(
            "/api/tasks/task_src/source",
            files={"file": ("source.pdf", pdf_bytes, "application/pdf")},
        )
        self.assertEqual(upload_response.status_code, 200)
        self.assertEqual(upload_response.json()["url"], "/api/tasks/task_src/source")

        page_response = self.client.get("/api/tasks/task_src/source/pages?start_page=2&end_page=3")
        self.assertEqual(page_response.status_code, 200)
        subset = PdfReader(io.BytesIO(page_response.content))
        self.assertEqual(len(subset.pages), 2)

    def test_task_save_strips_heavy_fields_when_external_source_exists(self):
        self.client.post(
            "/api/tasks/task_big/source",
            files={"file": ("source.pdf", b"%PDF-1.4\n", "application/pdf")},
        )
        task = {
            "id": "task_big",
            "name": "big.pdf",
            "sourceKind": "pdf",
            "sourceUrl": "/api/tasks/task_big/source",
            "sourceDataUrl": "data:application/pdf;base64," + ("x" * 1000),
            "batches": [
                {
                    "id": "b1",
                    "status": "pending",
                    "pageCount": 20,
                    "payloadDataUrl": "data:application/pdf;base64," + ("y" * 1000),
                }
            ],
        }

        response = self.client.put("/api/tasks/task_big", json=task)
        self.assertEqual(response.status_code, 200)

        detail = self.client.get("/api/tasks/task_big").json()
        self.assertEqual(detail["sourceUrl"], "/api/tasks/task_big/source")
        self.assertNotIn("sourceDataUrl", detail)
        self.assertNotIn("payloadDataUrl", detail["batches"][0])

    def test_task_save_splits_results_into_sidecar_and_preserves_them_on_metadata_save(self):
        task = {
            "id": "task_side",
            "name": "sidecar.pdf",
            "sourceKind": "pdf",
            "status": "processing",
            "pageCount": 1,
            "batches": [
                {"id": "b1", "status": "completed", "pageCount": 1, "markdown": "Batch text"}
            ],
            "markdown": "# Heavy Markdown",
            "images": {"ocr_images/a.jpg": "base64-image"},
            "ocrResults": [{"markdown": {"text": "# Heavy Markdown"}}],
        }

        response = self.client.put("/api/tasks/task_side", json=task)
        self.assertEqual(response.status_code, 200)

        task_path = Path(self.temp_dir.name) / "task_side" / "task.json"
        result_path = Path(self.temp_dir.name) / "task_side" / "result.json"
        stored = json.loads(task_path.read_text(encoding="utf-8"))
        self.assertNotIn("markdown", stored)
        self.assertNotIn("images", stored)
        self.assertNotIn("ocrResults", stored)
        self.assertTrue(result_path.exists())

        metadata_only = {
            "id": "task_side",
            "name": "sidecar.pdf",
            "sourceKind": "pdf",
            "status": "completed",
            "pageCount": 1,
            "batches": [{"id": "b1", "status": "completed", "pageCount": 1}],
            "_preserveResult": True,
        }
        response = self.client.put("/api/tasks/task_side", json=metadata_only)
        self.assertEqual(response.status_code, 200)

        detail = self.client.get("/api/tasks/task_side").json()
        self.assertEqual(detail["markdown"], "# Heavy Markdown")
        self.assertEqual(detail["images"], {"ocr_images/a.jpg": "base64-image"})
        self.assertEqual(detail["ocrResults"], [{"markdown": {"text": "# Heavy Markdown"}}])
        self.assertEqual(detail["batches"][0]["markdown"], "Batch text")

    def test_batch_markdown_only_task_is_marked_as_having_markdown(self):
        task = {
            "id": "task_batch_markdown",
            "name": "batch-only.pdf",
            "batches": [{"id": "b1", "status": "completed", "pageCount": 1, "markdown": "Batch text"}],
        }

        response = self.client.put("/api/tasks/task_batch_markdown", json=task)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["task"]["hasMarkdown"])

    def test_clear_tasks_only_removes_task_directories(self):
        task_dir = Path(self.temp_dir.name) / "task_keep"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.json").write_text('{"id":"task_keep"}', encoding="utf-8")
        keep_file = Path(self.temp_dir.name) / "keep.txt"
        keep_file.write_text("keep", encoding="utf-8")
        keep_dir = Path(self.temp_dir.name) / "docs"
        keep_dir.mkdir(exist_ok=True)

        response = self.client.delete("/api/tasks")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(task_dir.exists())
        self.assertTrue(keep_file.exists())
        self.assertTrue(keep_dir.exists())

    def test_model_runtime_switch_is_rejected_while_ocr_is_active(self):
        self.server.ocr_active_count = 1
        try:
            with patch.object(self.server, "model_control_available", return_value=True):
                response = self.client.post("/api/model-runtime/switch", json={"modelId": "pp-ocrv6"})
            self.assertEqual(response.status_code, 409)
        finally:
            self.server.ocr_active_count = 0

    def test_ocr_request_is_rejected_during_model_switch(self):
        self.server.set_model_runtime_operation("switching", "Switching to pp-ocrv6", "pp-ocrv6")
        try:
            response = self.client.post(
                "/api/paddleocr-vl-1.6",
                json={"image": "AA==", "fileType": 1},
            )
            self.assertEqual(response.status_code, 409)
        finally:
            self.server.set_model_runtime_operation("idle", "", "paddleocr-vl-1.6")


if __name__ == "__main__":
    unittest.main()
