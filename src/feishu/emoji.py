class EmojiType:
    GET = "OnIt"
    TYPING = "Typing"
    THINKING = "THINKING"
    ONE_SEC = "OneSecond"

    DONE = "DONE"
    CHECK_MARK = "DONE"
    LGTM = "LGTM"
    THUMBSUP = "THUMBSUP"
    OK = "OK"
    AWESOME = "AWESOME"

    CROSS_MARK = "SKULL"
    ERROR = "SKULL"
    SOB = "SOB"
    WRONG = "SOB"

    SMART = "FINGERHEART"
    ROCKET = "Fire"
    FIRE = "Fire"
    MUSCLE = "MUSCLE"
    YEAH = "YEAH"
    PARTY = "PARTY"

    SALUTE = "THANKS"
    HIGHFIVE = "FISTBUMP"
    WAVE = "WAVE"
    CLAP = "APPLAUSE"

    FLASH = "JIAYI"
    READING = "THINKING"
    BUSY = "OneSecond"
    LOCK = "LOCK"


class EmojiReaction:
    _ALLOWED_AUTO_REACTIONS = {
        EmojiType.TYPING,
        EmojiType.GET,
        EmojiType.PARTY,
    }

    @staticmethod
    def should_send(emoji_type: str) -> bool:
        """Return whether an automatic reaction should be sent to Feishu."""
        return emoji_type in EmojiReaction._ALLOWED_AUTO_REACTIONS

    @staticmethod
    def on_processing() -> str:
        return EmojiType.TYPING

    @staticmethod
    def on_error() -> str:
        return EmojiType.SOB

    @staticmethod
    def on_coco_enter() -> str:
        return EmojiType.SMART

    @staticmethod
    def on_coco_exit() -> str:
        return EmojiType.WAVE

    @staticmethod
    def on_multi_task_start() -> str:
        return EmojiType.ROCKET

    @staticmethod
    def on_multi_task_done() -> str:
        return EmojiType.PARTY

    @staticmethod
    def on_dir_changed() -> str:
        return EmojiType.CHECK_MARK

    @staticmethod
    def on_shell_executed() -> str:
        return EmojiType.DONE

    @staticmethod
    def on_smart_mode() -> str:
        return EmojiType.OK

    @staticmethod
    def on_coco_mode() -> str:
        return EmojiType.GET

    @staticmethod
    def on_chat_locked() -> str:
        return EmojiType.LOCK
