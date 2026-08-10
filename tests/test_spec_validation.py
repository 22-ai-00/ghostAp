import unittest
from unittest.mock import MagicMock

from pydantic import ValidationError

from src.spec_engine.engine import SpecEngine, SpecProjectStatus
from src.spec_engine.models import SpecProject
from src.spec_engine.validation import SpecInput


class TestSpecValidation(unittest.TestCase):
    def setUp(self):
        self.engine = SpecEngine(
            chat_id="test_chat",
            root_path="/tmp/test_project",
            agent_type="test_agent"
        )
        self.engine.settings = MagicMock()
        self.engine.settings.spec_max_cycles = 5

    def test_valid_input(self):
        # Should not raise exception
        input_data = SpecInput(requirement_text="Valid requirement", task_id="task_123")
        self.assertEqual(input_data.requirement_text, "Valid requirement")
        self.assertEqual(input_data.task_id, "task_123")

    def test_empty_requirement(self):
        with self.assertRaises(ValidationError):
            SpecInput(requirement_text="", task_id="task_123")

    def test_long_requirement(self):
        with self.assertRaises(ValidationError):
            SpecInput(requirement_text="a" * 50001, task_id="task_123")

    def test_engine_execute_validation_failure(self):
        callbacks = MagicMock()

        # Execute with empty requirement
        result = self.engine.execute(requirement_text="", callbacks=callbacks)

        # Verify the real project state machine reaches a terminal failure.
        self.assertIsInstance(result, SpecProject)
        self.assertEqual(result.status, SpecProjectStatus.FAILED)
        self.assertIn("非法配置参数", result.error)
        self.assertIsNotNone(result.completed_at)
        callbacks.on_error.assert_called_once_with(result.error)


if __name__ == "__main__":
    unittest.main()
