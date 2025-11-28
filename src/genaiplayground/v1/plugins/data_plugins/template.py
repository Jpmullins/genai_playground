from dataclasses import dataclass


@dataclass
class Template:
    user_template: str
    assistant_template: str
    system_template: str

    def render_message(self, message: "dict[str, str]") -> str:
        return self.user_template.format(**message)
