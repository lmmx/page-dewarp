__all__ = ["add_default_argument"]


def add_default_argument(
    parser,
    name_or_flags,
    arg_name=None,
    action=None,
    help=None,
    const=None,
    choices=None,
    nargs=None,
    metavar=None,
    required=None,
):
    kwargs = {
        "action": action,
        "nargs": nargs,
        "const": const,
        "choices": choices,
        "help": help,
        "metavar": metavar,
        "required": required,
    }
    if arg_name is None:
        title = name_or_flags if isinstance(name_or_flags, str) else name_or_flags[1]
        arg_name = title.lstrip("-").upper().replace("-", "_")
    default_value = parser.get_config_param(arg_name)
    default_type = type(default_value)
    if default_type is bool and action is const is None:
        kwargs["action"] = f"store_{'false' if default_value else 'true'}"
    else:
        kwargs["type"] = default_type
    if isinstance(name_or_flags, str):
        name_or_flags = [name_or_flags]  # will be star-expanded so must be sequence
    if kwargs["help"] is None:
        # comment in TOML as a string if present, or `None` if absent
        kwargs["help"] = parser.config_comments().get(arg_name)
    parser.add_argument(
        *name_or_flags,  # one or two values
        dest=arg_name,
        default=default_value,
        **{kw: v for kw, v in kwargs.items() if v is not None},
    )
