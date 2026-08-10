from __future__ import annotations

import unittest

from core.sd_prompt_bridge import extract_sd_prompt_block
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


if __name__ == "__main__":
    unittest.main()
