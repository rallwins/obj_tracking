import unittest

from fastapi.testclient import TestClient

from obj_tracking.server import create_app


class ServerFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def test_frontend_and_assets_are_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Tracking studio", response.text)
        self.assertEqual(self.client.get("/assets/styles.css").status_code, 200)
        script = self.client.get("/assets/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("System Settings", script.text)

    def test_health_endpoint_does_not_load_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_loaded": False})

    def test_upload_rejects_unsupported_file_type(self):
        response = self.client.post(
            "/track",
            files={"video": ("notes.txt", b"not a video", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
