from __future__ import annotations

import unittest

from core.sd_prompt_bridge import extract_pose_prompt_block, extract_sd_and_pose_prompts, extract_sd_prompt_block
from grok_bridge.stream_parser import GrokStreamParser


class SdPromptMultipleTagTests(unittest.TestCase):
    def test_extracts_second_configured_pair(self) -> None:
        clean, prompt = extract_sd_prompt_block(
            "返答です。\n$ BEGIN $\nmasterpiece, portrait\n$ END $",
            begin_tag="[BEGIN] | $ BEGIN $",
            end_tag="[END] | $ END $",
        )
        self.assertEqual(clean, "返答です。")
        self.assertEqual(prompt, "masterpiece, portrait")

    def test_stream_parser_uses_matching_second_end_tag(self) -> None:
        spoken: list[str] = []
        prompts: list[str] = []
        parser = GrokStreamParser(
            spoken.append,
            prompts.append,
            begin_tag="[BEGIN] | $ BEGIN $",
            end_tag="[END] | $ END $",
        )
        parser.feed("返答。\n$ BEGIN $\nportrait", final=False)
        parser.feed("返答。\n$ BEGIN $\nportrait\n$ END $", final=True)
        self.assertEqual(prompts, ["portrait"])
        self.assertNotIn("$ BEGIN $", "".join(spoken))

    def test_pose_block_is_joined_to_sd_prompt(self) -> None:
        text = (
            "===SD_BEGIN===\nportrait, lying on bed\n===SD_END===\n"
            "===POSE_BEGIN===\nlying on back with legs spread\n===POSE_END==="
        )
        clean, prompt = extract_sd_prompt_block(
            text,
            begin_tag="===SD_BEGIN===",
            end_tag="===SD_END===",
        )
        self.assertEqual(clean, "")
        self.assertEqual(prompt, "portrait, lying on bed,\nlying on back with legs spread")

    def test_stream_parser_sends_sd_and_pose_once_combined(self) -> None:
        spoken: list[str] = []
        prompts: list[str] = []
        parser = GrokStreamParser(
            spoken.append,
            prompts.append,
            begin_tag="===SD_BEGIN===",
            end_tag="===SD_END===",
        )
        text = (
            "===SD_BEGIN===portrait===SD_END==="
            "===POSE_BEGIN===lying on back===POSE_END==="
        )
        parser.feed(text, final=True)
        self.assertEqual(prompts, ["portrait,\nlying on back"])
        self.assertEqual(spoken, [])

    def test_pose_can_be_extracted_separately(self) -> None:
        clean, pose = extract_pose_prompt_block(
            "会話\n===POSE_BEGIN===lying on back===POSE_END==="
        )
        self.assertEqual(clean, "会話")
        self.assertEqual(pose, "lying on back")

    def test_sd_and_pose_are_available_independently(self) -> None:
        clean, sd_prompt, pose_prompt = extract_sd_and_pose_prompts(
            "返答\n===SD_BEGIN===portrait===SD_END==="
            "===POSE_BEGIN===kneeling pose===POSE_END===",
            sd_begin_tag="===SD_BEGIN===",
            sd_end_tag="===SD_END===",
        )
        self.assertEqual(clean, "返答")
        self.assertEqual(sd_prompt, "portrait")
        self.assertEqual(pose_prompt, "kneeling pose")


if __name__ == "__main__":
    unittest.main()
