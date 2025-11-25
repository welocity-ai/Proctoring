
import unittest
from unittest.mock import MagicMock, patch
import time
import sys
import os

# Add parent dir to path to import vimeo_proctor_report
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vimeo_proctor_report import main
import vimeo_proctor_report

class TestParallelExecution(unittest.TestCase):

    @patch('vimeo_proctor_report.analyze_face_and_gaze')
    @patch('vimeo_proctor_report.run_object_detection')
    @patch('vimeo_proctor_report.run_diarization_and_extract_snippets')
    @patch('vimeo_proctor_report.extract_audio')
    @patch('vimeo_proctor_report.build_pdf_report')
    @patch('vimeo_proctor_report.ProcessPoolExecutor')
    def test_parallel_execution(self, mock_executor_cls, mock_build, mock_extract, mock_voice, mock_obj, mock_face):
        
        # Mock cv2
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value.get.side_effect = [30.0, 300.0] # fps, frame_count
        
        with patch.dict(sys.modules, {'cv2': mock_cv2}):
            # Setup mocks
            mock_extract.return_value = "dummy_audio.wav"
            
            # Create real Futures to simulate async completion
            from concurrent.futures import Future
            f_face = Future()
            f_obj = Future()
            f_voice = Future()
            
            # Set results
            f_face.set_result({"face_flag_logs": []})
            f_obj.set_result(([], 0))
            f_voice.set_result(([], 0))
            
            # Configure executor mock
            mock_executor = mock_executor_cls.return_value
            mock_executor.__enter__.return_value = mock_executor
            
            # We need to map submit calls to specific futures based on the function passed
            def submit_side_effect(func, *args, **kwargs):
                if func == mock_face:
                    return f_face
                elif func == mock_obj:
                    return f_obj
                elif func == mock_voice:
                    return f_voice
                return Future()
                
            mock_executor.submit.side_effect = submit_side_effect

            # Mock args
            test_args = [
                "vimeo_proctor_report.py",
                "dummy_video.mp4",
                "--hf-token", "dummy_token",
                "--outdir", "tests/out"
            ]
            
            with patch.object(sys, 'argv', test_args):
                vimeo_proctor_report.main()
            
            # Verify calls
            self.assertEqual(mock_executor.submit.call_count, 3)
            
            # Verify specific calls
            # Note: We can't easily assert exact call args because of the dynamic nature, 
            # but we know side_effect worked if the script finished without error.
            
            print("Test finished successfully")

if __name__ == '__main__':
    unittest.main()
