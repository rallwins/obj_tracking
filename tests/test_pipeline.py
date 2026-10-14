import unittest
from types import SimpleNamespace

from obj_tracking.pipeline import detections_from_result, parse_source


class TensorLike:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class PipelineTests(unittest.TestCase):
    def test_parse_source_converts_camera_index(self):
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("12"), 12)
        self.assertEqual(parse_source("video.mp4"), "video.mp4")
        self.assertEqual(parse_source("-1"), "-1")

    def test_detections_from_result_converts_xyxy_to_xywh(self):
        box = SimpleNamespace(
            xyxy=[TensorLike([10, 20, 50, 80])],
            conf=[TensorLike(0.9)],
            cls=[TensorLike(1)],
        )
        result = SimpleNamespace(boxes=[box])

        detections = detections_from_result(result, ["person", "car"])

        self.assertEqual(detections, [([10.0, 20.0, 40.0, 60.0], 0.9, "car")])


if __name__ == "__main__":
    unittest.main()
