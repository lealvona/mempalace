"""TDD: save hook must actually mine conversations without MEMPAL_DIR.

The save hook should auto-discover the conversation transcript and mine it
without the user needing to set MEMPAL_DIR. Currently MEMPAL_DIR defaults
to empty, which means the mining block is skipped and nothing is saved
despite the hook telling the agent "saved in background."

Written BEFORE the fix.
"""

import os


class TestSaveHookAutoMines:
    """The save hook must mine the active transcript automatically."""

    def test_hook_mines_transcript_path(self):
        """The hook receives TRANSCRIPT_PATH from Claude Code.
        It should use that to mine the conversation as --mode convos,
        independently of MEMPAL_DIR (which is for project files only)."""
        hook_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hooks",
            "mempal_save_hook.sh",
        )
        src = open(hook_path).read()

        # The hook must drive the conversation mine off TRANSCRIPT_PATH,
        # using `dirname` to derive the parent dir, and tagging it with
        # `--mode convos` so the convo miner runs (not the projects miner).
        assert "TRANSCRIPT_PATH" in src, "hook must read transcript_path"
        assert "mempalace mine" in src, "hook must invoke `mempalace mine`"
        assert (
            'dirname "$TRANSCRIPT_PATH"' in src
        ), "hook must mine the transcript's parent directory"
        assert (
            "--mode convos" in src
        ), "transcript mine must use --mode convos, not the projects miner"

    def test_mempal_dir_default_not_empty(self):
        """If MEMPAL_DIR is still used, it should have a sensible default,
        not an empty string that silently disables mining."""
        hook_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hooks",
            "mempal_save_hook.sh",
        )
        src = open(hook_path).read()

        # Check if MEMPAL_DIR defaults to empty
        has_empty_default = 'MEMPAL_DIR=""' in src

        # If it defaults to empty, mining is silently disabled
        if has_empty_default:
            # There must be an alternative mining path that doesn't need MEMPAL_DIR
            has_alternative = (
                src.count("mempalace mine") > 1
                or "TRANSCRIPT_PATH" in src.split("mempalace mine")[0]
            )
            assert has_alternative, (
                'MEMPAL_DIR defaults to "" which silently disables mining. '
                "Either set a default path or add transcript-based mining."
            )
