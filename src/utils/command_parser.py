"""Small parser for the legacy Deep and Workflow command surfaces."""

from dataclasses import dataclass, field


@dataclass
class ParsedCommand:
    command: str
    args: str
    flags: dict[str, bool] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return bool(self.command)


class CommandParser:
    @staticmethod
    def parse(text: str, known_flags: set[str] | None = None) -> ParsedCommand:
        text = text.strip()
        if not text:
            return ParsedCommand(command="", args="")
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        if len(parts) == 1:
            return ParsedCommand(command=command, args="")
        raw_args = parts[1]
        if known_flags:
            extracted_flags: dict[str, bool] = {}
            filtered_tokens: list[str] = []
            for token in raw_args.split():
                flag_name = token.lstrip("-")
                if token.startswith("-") and flag_name in known_flags:
                    extracted_flags[flag_name] = True
                else:
                    filtered_tokens.append(token)
            return ParsedCommand(
                command=command,
                args=" ".join(filtered_tokens),
                flags=extracted_flags,
            )
        return ParsedCommand(command=command, args=raw_args)

    @staticmethod
    def parse_basic(text: str) -> ParsedCommand:
        text = text.strip()
        if not text:
            return ParsedCommand(command="", args="")
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        tokens = args.lower().strip().split()
        flags = {token.lstrip("-"): True for token in tokens} if tokens and all(token.startswith("-") for token in tokens) else {}
        return ParsedCommand(command=command, args=args, flags=flags)
