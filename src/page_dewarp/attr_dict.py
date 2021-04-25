class AttrDict(dict):
    # Core attribute logic adapted from util of the same name in fastcore
    "`dict` subclass that also provides access to keys as attrs"

    def __getattr__(self, k):
        return self[k] if k in self else stop(AttributeError(k))

    def __setattr__(self, k, v):
        (super().__setattr__ if k.startswith("_") else self.__setitem__)(k, v)

    def __dir__(self):
        return super().__dir__() + list(self.keys())

    @classmethod
    def from_dict(cls, d):
        "Convert (possibly nested) dicts (or lists of dicts) to `AttrDict`"
        return cls.convert_dict_format(d, to_dict=False)

    @classmethod
    def as_dict(cls, d):
        "Convert (possibly nested) AttrDicts (or lists of AttrDicts) to `dict`"
        return cls.convert_dict_format(d, to_dict=True)

    @classmethod
    def convert_dict_format(cls, d, to_dict=False):
        convert = cls.as_dict if to_dict else cls.from_dict
        out_fmt = dict if to_dict else AttrDict
        if isinstance(d, dict):
            out = out_fmt(**{k: convert(v) for k, v in d.items()})
        else:
            out = d  # nothing to do
        return out


def stop(e):
    "Wrap raise statement as an expression for use with ternary operator"
    raise e
