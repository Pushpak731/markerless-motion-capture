import unittest

from src.frame_quality import FrameQualityAnalyzer, interpolate_landmarks


def make_landmarks(x_offset=0.0, visibility=1.0):
    landmarks = []
    for idx in range(33):
        landmarks.append({
            'x': x_offset + (idx * 0.01),
            'y': 0.1 + (idx * 0.005),
            'z': 0.01 * idx,
            'v': visibility,
        })
    return landmarks


class FrameQualityTests(unittest.TestCase):
    def test_classifies_valid_and_missing_frames(self):
        analyzer = FrameQualityAnalyzer()

        missing = analyzer.analyze([], 640, 480)
        self.assertFalse(missing.pose_detected)
        self.assertEqual(missing.reason, 'missing')

        valid = analyzer.analyze(make_landmarks(), 640, 480)
        self.assertTrue(valid.pose_detected)
        self.assertTrue(valid.pose_usable)
        self.assertEqual(valid.reason, 'valid')

    def test_classifies_low_confidence_partial_and_unstable(self):
        analyzer = FrameQualityAnalyzer(unstable_jump_threshold=0.05)

        low_confidence = analyzer.analyze(make_landmarks(visibility=0.2), 640, 480)
        self.assertEqual(low_confidence.reason, 'low_confidence')
        self.assertFalse(low_confidence.pose_usable)

        partial = analyzer.analyze(make_landmarks(), 640, 480)
        partial_landmarks = make_landmarks()
        for idx in range(20, 33):
            partial_landmarks[idx]['v'] = 0.1
        partial = analyzer.analyze(partial_landmarks, 640, 480)
        self.assertEqual(partial.reason, 'partial_pose')
        self.assertFalse(partial.pose_usable)

        analyzer = FrameQualityAnalyzer(unstable_jump_threshold=0.01)
        analyzer.analyze(make_landmarks(), 640, 480)
        unstable = analyzer.analyze(make_landmarks(x_offset=1.0), 640, 480)
        self.assertEqual(unstable.reason, 'unstable')
        self.assertFalse(unstable.pose_usable)

    def test_interpolates_landmarks_linearly(self):
        start = make_landmarks(x_offset=0.0)
        end = make_landmarks(x_offset=1.0)

        interpolated = interpolate_landmarks(start, end, 0.25)

        self.assertAlmostEqual(interpolated[0]['x'], 0.25, places=6)
        self.assertAlmostEqual(interpolated[10]['x'], 0.35, places=6)

    def test_summary_includes_warnings_and_recommendations(self):
        analyzer = FrameQualityAnalyzer()
        analyzer.analyze(make_landmarks(), 640, 480)
        analyzer.analyze([], 640, 480)
        analyzer.analyze([], 640, 480)
        analyzer.analyze(make_landmarks(visibility=0.2), 640, 480)

        summary = analyzer.build_summary(total_frames=4, interpolated_frame_count=1, tracking_loss_frame_count=1)

        self.assertIn('input_quality_warnings', summary)
        self.assertIn('recommendations', summary)
        self.assertIn('detected_pose_coverage', summary)
        self.assertIn('usable_pose_coverage', summary)
        self.assertGreaterEqual(summary['missing_frame_count'], 2)
        self.assertGreaterEqual(summary['low_confidence_frame_count'], 1)


if __name__ == '__main__':
    unittest.main()